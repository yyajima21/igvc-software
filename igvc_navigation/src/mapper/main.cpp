// Subscribes to Point Cloud Data, updates the occupancy grid, then publishes the data.

#include <cv_bridge/cv_bridge.h>
#include <igvc_msgs/map.h>
#include <nav_msgs/Odometry.h>
#include <pcl/point_cloud.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <igvc_utils/RobotState.hpp>
#include <vector>
#include <unordered_set>

cv_bridge::CvImage img_bridge;

ros::Publisher map_pub;
ros::Publisher map_empty_pub;
ros::Publisher debug_pub;
ros::Publisher debug_pcl_pub;
ros::Publisher debug_empty_pcl_pub;
ros::Publisher debug_log_odds_pcl_pub;
std::unique_ptr<cv::Mat> published_map;  // matrix will be publishing
std::unique_ptr<cv::Mat> empty_map;  // empty points for debugging
std::unique_ptr<cv::Mat> line_map;  // probability matrix for lines
std::unique_ptr<cv::Mat> barrel_map;  // probability matrix for barrels
std::unique_ptr<Eigen::MatrixXf> log_odds_map_lines;  // log_odds map for lines
std::unique_ptr<Eigen::MatrixXf> log_odds_map_barrels;  // log_odds map for barrels
std::map<std::string, tf::StampedTransform> transforms;
std::unique_ptr<tf::TransformListener> tf_listener;

double resolution;
double transform_max_wait_time;
int start_x;  // start x location
int start_y;  // start y location
int length_y;
int width_x;
uchar occupancy_grid_threshold;
int increment_step;
bool debug;
double sensor_model_padding;

float toLogOdds(float px);
const float p_empty_given_empty = 0.55;
const float p_occ_given_occ = 0.55;
const float log_odds_empty = toLogOdds(1 - p_empty_given_empty);
const float log_odds_occupied = toLogOdds(p_occ_given_occ);
RobotState state;

std::tuple<double, double> rotate(double x, double y) {
  double newX = x * cos(state.yaw) - y * sin(state.yaw);
  double newY = x * sin(state.yaw) + y * cos(state.yaw);
  return (std::make_tuple(newX, newY));
}
float toLogOdds(float px) {
  return log(px/(1-px));
}

float fromLogOdds(float lx) {
  return 1 - (1/(exp(lx)));
}

uchar toUChar(float px) {
  return static_cast<uchar>(std::round(255*px));
}

void getOdomTransform(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &msg) {
  tf::StampedTransform transform;
  ros::Time messageTimeStamp;
  pcl_conversions::fromPCL(msg->header.stamp, messageTimeStamp);
  try {
    tf_listener->waitForTransform("/odom", "/base_link", messageTimeStamp, ros::Duration(transform_max_wait_time));
    tf_listener->lookupTransform("/odom", "/base_link", messageTimeStamp, transform);
  }
  catch (tf::TransformException &ex) {
    ROS_ERROR("%s", ex.what());
  }
  state.setState(transform);
}

void setMsgValues(igvc_msgs::map &message, sensor_msgs::Image &image, uint64_t pcl_stamp) {
  pcl_conversions::fromPCL(pcl_stamp, image.header.stamp);
  pcl_conversions::fromPCL(pcl_stamp, message.header.stamp);
  message.header.frame_id = "/odom";
  message.image = image;
  message.length = length_y;
  message.width = width_x;
  message.resolution = resolution;
  message.orientation = state.yaw;
  message.x = std::round(state.x / resolution) + start_x;
  message.y = std::round(state.y / resolution) + start_y;
  message.x_initial = start_x;
  message.y_initial = start_y;
}

template<typename T>
struct matrix_hash : std::unary_function<T, size_t> {
    std::size_t operator()(T const& matrix) const {
      // Note that it is oblivious to the storage order of Eigen matrix (column- or
      // row-major). It will give you the same hash value for two different matrices if they
      // are the transpose of each other in different storage order.
      size_t seed = 0;
      for (size_t i = 0; i < matrix.size(); ++i) {
        auto elem = *(matrix.data() + i);
        seed ^= std::hash<typename T::Scalar>()(elem) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
      }
      return seed;
    }
};

void invSensor(float x1, float y1, float x2, float y2, std::vector<Eigen::Vector2i> &vector,
                   std::unordered_set<Eigen::Vector2i, matrix_hash<Eigen::Vector2i>> &occupied_set, bool line) {
  const bool steep = (fabs(y2 - y1) > fabs(x2 - x1));
  if (steep) {
    std::swap(x1, y1);
    std::swap(x2, y2);
  }

  if (x1 > x2) {
    std::swap(x1, x2);
    std::swap(y1, y2);
  }

  const float dx = x2 - x1;
  const float dy = fabs(y2 - y1);

  float error = dx / 2.0f;
  const int ystep = (y1 < y2) ? 1 : -1;
  int y = (int) y1;

  const int maxX = (int) x2;

  for (int x = (int) x1; x < maxX; x++) {
    if (steep) {
      Eigen::Vector2i vec(y, x);
      if (occupied_set.find(vec) != occupied_set.end()) {
        break;
      }
      if (line && barrel_map->at<uchar>(y, x) > (uchar) 230) {
        break;
      }
      vector.emplace_back(vec);
    } else {
      Eigen::Vector2i vec(x, y);
      if (occupied_set.find(vec) != occupied_set.end()) {
        break;
      }
      if (line && barrel_map->at<uchar>(x, y) > (uchar) 230) {
        break;
      }
      vector.emplace_back(vec);
    }

    error -= dy;
    if (error < 0) {
      y += ystep;
      error += dx;
    }
  }
}
// TODO: Update map when you update the cell
void updateGridBarrels(const pcl::PointCloud<pcl::PointXYZ>::Ptr &transformed) {
  pcl::PointCloud<pcl::PointXYZ>::const_iterator point_iter;

  std::vector<Eigen::Vector2i> empty_points;
  std::unordered_set<Eigen::Vector2i, matrix_hash<Eigen::Vector2i>> occupied_set;
  empty_points.reserve(255);

  // First pass to put all points into set
  for (point_iter = transformed->begin(); point_iter < transformed->points.end(); point_iter++) {
    double x_point_raw, y_point_raw;
    std::tie(x_point_raw, y_point_raw) = rotate(point_iter->x, point_iter->y);

    int point_x = static_cast<int>(std::round(x_point_raw / resolution + state.x / resolution + start_x));
    int point_y = static_cast<int>(std::round(y_point_raw / resolution + state.y / resolution + start_y));
    occupied_set.emplace(Eigen::Vector2i(point_x, point_y));
  }

  for (point_iter = transformed->begin(); point_iter < transformed->points.end(); point_iter++) {
    double x_point_raw, y_point_raw;
    std::tie(x_point_raw, y_point_raw) = rotate(point_iter->x, point_iter->y);

    int point_x = static_cast<int>(std::round(x_point_raw / resolution + state.x / resolution + start_x));
    int point_y = static_cast<int>(std::round(y_point_raw / resolution + state.y / resolution + start_y));
    int robot_grid_x = static_cast<int>(std::round(state.x / resolution + start_x));
    int robot_grid_y = static_cast<int>(std::round(state.y / resolution + start_x));

    double angle = atan2(y_point_raw, x_point_raw);
    double magnitude = sqrt(x_point_raw*x_point_raw + y_point_raw*y_point_raw) - sensor_model_padding;
    if (magnitude > 0) {
      int clear_x = static_cast<int>(std::round((cos(angle)*magnitude) / resolution + state.x / resolution + start_x));
      int clear_y = static_cast<int>(std::round((sin(angle)*magnitude) / resolution + state.y / resolution + start_y));

      invSensor(robot_grid_x, robot_grid_y, clear_x, clear_y, empty_points, occupied_set, false);
      for (auto cell : empty_points) {
        if (!(std::tie(cell.x(), cell.y()) == std::tie(robot_grid_x, robot_grid_y) ||
              std::tie(cell.x(), cell.y()) == std::tie(point_x, point_y))) {
          (*log_odds_map_barrels)(cell.y(), cell.x()) += log_odds_empty;
        }
      }
    }
  }
}

void updateGridLines(const pcl::PointCloud<pcl::PointXYZ>::Ptr &transformed) {
  pcl::PointCloud<pcl::PointXYZ>::const_iterator point_iter;

  std::vector<Eigen::Vector2i> empty_points;
  std::unordered_set<Eigen::Vector2i, matrix_hash<Eigen::Vector2i>> occupied_set;
  empty_points.reserve(255);
  // Clear empty map testing
  *empty_map = cv::Mat::zeros(empty_map->size(), empty_map->type());

  // First pass to put all points into a set
  for (point_iter = transformed->begin(); point_iter < transformed->points.end(); point_iter++) {
    double x_point_raw, y_point_raw;
    std::tie(x_point_raw, y_point_raw) = rotate(point_iter->x, point_iter->y);

    int point_x = static_cast<int>(std::round(x_point_raw / resolution + state.x / resolution + start_x));
    int point_y = static_cast<int>(std::round(y_point_raw / resolution + state.y / resolution + start_y));
    occupied_set.emplace(Eigen::Vector2i(point_x, point_y));
  }

  for (point_iter = transformed->begin(); point_iter < transformed->points.end(); point_iter++) {
    double x_point_raw, y_point_raw;
    std::tie(x_point_raw, y_point_raw) = rotate(point_iter->x, point_iter->y);

    int point_x = static_cast<int>(std::round(x_point_raw / resolution + state.x / resolution + start_x));
    int point_y = static_cast<int>(std::round(y_point_raw / resolution + state.y / resolution + start_y));
    int robot_grid_x = static_cast<int>(std::round(state.x / resolution + start_x));
    int robot_grid_y = static_cast<int>(std::round(state.y / resolution + start_x));

    double angle = atan2(y_point_raw, x_point_raw);
    double magnitude = sqrt(x_point_raw*x_point_raw + y_point_raw*y_point_raw) - sensor_model_padding;
    if (magnitude > 0) {
      int clear_x = static_cast<int>(std::round((cos(angle)*magnitude) / resolution + state.x / resolution + start_x));
      int clear_y = static_cast<int>(std::round((sin(angle)*magnitude) / resolution + state.y / resolution + start_y));

      invSensor(robot_grid_x, robot_grid_y, clear_x, clear_y, empty_points, occupied_set, true);
      for (auto cell : empty_points) {
        if (!(std::tie(cell.x(), cell.y()) == std::tie(robot_grid_x, robot_grid_y) ||
              std::tie(cell.x(), cell.y()) == std::tie(point_x, point_y))) {
          empty_map->at<uchar>(cell.x(), cell.y()) = 1;
          if (published_map->at<uchar>(cell.x(), cell.y()) > 5) {
            published_map->at<uchar>(cell.x(), cell.y()) -= (uchar) 5;
          }
          (*log_odds_map_lines)(cell.y(), cell.x()) += log_odds_empty;
        }
      }
    }
  }
}
void updateOccupancyGrid(const pcl::PointCloud<pcl::PointXYZ>::Ptr &transformed) {
  int offMapCount = 0;

  pcl::PointCloud<pcl::PointXYZ>::const_iterator point_iter;

  for (point_iter = transformed->begin(); point_iter < transformed->points.end(); point_iter++) {
    double x_point_raw, y_point_raw;
    std::tie(x_point_raw, y_point_raw) = rotate(point_iter->x, point_iter->y);

    int point_x = static_cast<int>(std::round(x_point_raw / resolution + state.x / resolution + start_x));
    int point_y = static_cast<int>(std::round(y_point_raw / resolution + state.y / resolution + start_y));

    if (point_x >= 0 && point_y >= 0 && point_x < length_y && start_y < width_x) {
      if (published_map->at<uchar>(point_x, point_y) <= UCHAR_MAX - (uchar) increment_step) {
        published_map->at<uchar>(point_x, point_y) += (uchar) increment_step;
      } else {
        published_map->at<uchar>(point_x, point_y) = UCHAR_MAX;
      }
      (*log_odds_map_lines)(point_y, point_x) += log_odds_occupied;
      line_map->at<uchar>(point_x, point_y) = (uchar) toLogOdds((*log_odds_map_lines)(point_y, point_x));
    } else {
      offMapCount++;
    }
  }
  if (offMapCount > 0) {
    ROS_WARN_STREAM(offMapCount << " points were off the map");
  }
}

void checkExistsStaticTransform(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &msg, const std::string &topic) {
  if (transforms.find(topic) == transforms.end()) {
    // Wait for transform between frame_id (ex. /scan/pointcloud) and base_footprint.
    ros::Time messageTimeStamp;
    pcl_conversions::fromPCL(msg->header.stamp, messageTimeStamp);
    if (tf_listener->waitForTransform("/base_footprint", msg->header.frame_id, messageTimeStamp, ros::Duration(3.0))) {
      ROS_INFO_STREAM("\n\ngetting transform for " << topic << "\n\n");
      tf::StampedTransform transform;
      tf_listener->lookupTransform("/base_footprint", msg->header.frame_id, messageTimeStamp, transform);
      transforms.insert(std::pair<std::string, tf::StampedTransform>(topic, transform));
    } else {
      ROS_ERROR_STREAM("\n\nfailed to find transform using empty transform\n\n");
    }
  }
}

void decayMap(const ros::TimerEvent &) {
  int nRows = published_map->rows;
  int nCols = published_map->cols;

  if (published_map->isContinuous()) {
    nCols *= nRows;
    nRows = 1;
  }
  int i, j;
  uchar *p;
  for (i = 0; i < nRows; i++) {
    p = published_map->ptr<uchar>(i);
    for (j = 0; j < nCols; j++) {
      if (p[j] > 0) {
        p[j] -= (uchar) 1;
      }
    }
  }
}

void frame_callback(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &msg, const std::string &topic) {
  // transform pointcloud into the occupancy grid, no filtering right now

  // make transformed clouds
  pcl::PointCloud<pcl::PointXYZ>::Ptr transformed =
    pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());

  // Check if static transform already exists for this topic.
  checkExistsStaticTransform(msg, topic);

  // Lookup transform form Ros Localization for position
  getOdomTransform(msg);

  // Apply transformation to msg points using the transform for this topic.
  pcl_ros::transformPointCloud(*msg, *transformed, transforms.at(topic));
  updateOccupancyGrid(transformed);
  if (topic == "/semantic_segmentation_cloud") {
    updateGridLines(transformed);
  } else {
    updateGridBarrels(transformed);
  }

  igvc_msgs::map message;    // >> message to be sent
  sensor_msgs::Image image;  // >> image in the message
  img_bridge = cv_bridge::CvImage(message.header, sensor_msgs::image_encodings::MONO8, *published_map);
  img_bridge.toImageMsg(image);  // from cv_bridge to sensor_msgs::Image

  // TODO: ======================================
  igvc_msgs::map message_empty;  // >> message to be sent
  sensor_msgs::Image image_empty;  // >> image in the message
  img_bridge = cv_bridge::CvImage(message.header, sensor_msgs::image_encodings::MONO8, *empty_map);
  img_bridge.toImageMsg(image_empty);  // from cv_bridge to sensor_msgs::Image

  setMsgValues(message_empty, image_empty, msg->header.stamp);
  map_empty_pub.publish(message_empty);
  // ============================================

  setMsgValues(message, image, msg->header.stamp);
  map_pub.publish(message);
  if (debug) {
    debug_pub.publish(image);
    // ROS_INFO_STREAM("\nThe robot is located at " << state);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr fromOcuGrid =
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>());
    for (int i = 0; i < width_x; i++) {
      for (int j = 0; j < length_y; j++) {
        if (published_map->at<uchar>(i, j) >= occupancy_grid_threshold) {
          // Set x y coordinates as the center of the grid cell.
          pcl::PointXYZRGB p(255, published_map->at<uchar>(i, j), published_map->at<uchar>(i, j));
          p.x = (i - start_x) * resolution;
          p.y = (j - start_y) * resolution;
          fromOcuGrid->points.push_back(p);
        }
      }
    }
    fromOcuGrid->header.frame_id = "/odom";
    fromOcuGrid->header.stamp = msg->header.stamp;
    debug_pcl_pub.publish(fromOcuGrid);

    // DEBUG ======================================
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr ocu_pcl =
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>());
    for (int i = 0; i < width_x; i++) {
      for (int j = 0; j < length_y; j++) {
        if (empty_map->at<uchar>(i, j) == 1) {
          // Set x y coordinates as the center of the grid cell.
          pcl::PointXYZRGB p(255, empty_map->at<uchar>(i, j), empty_map->at<uchar>(i, j));
          p.x = (i - start_x) * resolution;
          p.y = (j - start_y) * resolution;
          ocu_pcl->points.emplace_back(p);
        }
      }
    }
    ocu_pcl->header.frame_id = "/odom";
    ocu_pcl->header.stamp = msg->header.stamp;
    debug_empty_pcl_pub.publish(ocu_pcl);


    pcl::PointCloud<pcl::PointXYZRGB>::Ptr occupancy_map_pcl =
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>());
    for (size_t i = 0, size = log_odds_map_lines->size(); i < size; i++)
    {
      int y = static_cast<int>(i % log_odds_map_lines->rows());
      int x = static_cast<int>(i / log_odds_map_lines->rows());
      if (*(log_odds_map_lines->data() + i) != 0 || *(log_odds_map_barrels->data() + i) != 0) {
        // Eigen::Matrix is column major
        float p_line = fromLogOdds(*(log_odds_map_lines->data() + i));
        float p_barrel = fromLogOdds(*(log_odds_map_barrels->data() + i));
        float p_occupied = p_line + p_barrel > 1 ? 1 : p_line + p_barrel;
        uchar uCharVal = toUChar(p_occupied);
        pcl::PointXYZRGB p(uCharVal, uCharVal, uCharVal);
        p.x = (x - start_x) * resolution;
        p.y = (y - start_y) * resolution;
        occupancy_map_pcl->points.emplace_back(p);
      }
    }
    occupancy_map_pcl->header.frame_id = "/odom";
    occupancy_map_pcl->header.stamp = msg->header.stamp;
    debug_log_odds_pcl_pub.publish(occupancy_map_pcl);
    // DEBUG ========================================
  }
  //  update = true;
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "mapper");
  ros::NodeHandle nh;
  ros::NodeHandle pNh("~");
  std::string topics;

  std::list<ros::Subscriber> subs;
  tf_listener = std::unique_ptr<tf::TransformListener>(new tf::TransformListener());

  double cont_start_x;
  double cont_start_y;
  double decay_period;
  int cont_occupancy_grid_threshold;

  if (!(pNh.hasParam("topics") && pNh.hasParam("occupancy_grid_width") && pNh.hasParam("occupancy_grid_length") &&
        pNh.hasParam("occupancy_grid_resolution") && pNh.hasParam("start_X") && pNh.hasParam("start_Y") &&
        pNh.hasParam("increment_step") && pNh.hasParam("occupancy_grid_threshold") && pNh.hasParam("decay_period") &&
        pNh.hasParam("transform_max_wait_time") && pNh.hasParam("debug"))) {
    ROS_ERROR_STREAM("missing parameters; exiting");
    return 0;
  }

  // assumes all params inputted in meters
  pNh.getParam("topics", topics);
  pNh.getParam("occupancy_grid_length", length_y);
  pNh.getParam("occupancy_grid_width", width_x);
  pNh.getParam("occupancy_grid_resolution", resolution);
  pNh.getParam("occupancy_grid_threshold", cont_occupancy_grid_threshold);
  pNh.getParam("decay_period", decay_period);
  pNh.getParam("transform_max_wait_time", transform_max_wait_time);
  pNh.getParam("start_X", cont_start_x);
  pNh.getParam("start_Y", cont_start_y);
  pNh.getParam("increment_step", increment_step);
  pNh.getParam("debug", debug);
  pNh.getParam("sensor_model_padding", sensor_model_padding);

  // convert from meters to grid
  length_y = static_cast<int>(std::round(length_y / resolution));
  width_x = static_cast<int>(std::round(width_x / resolution));
  start_x = static_cast<int>(std::round(cont_start_x / resolution));
  start_y = static_cast<int>(std::round(cont_start_y / resolution));
  occupancy_grid_threshold = static_cast<uchar>(cont_occupancy_grid_threshold);
  ROS_INFO_STREAM("cv::Mat length: " << length_y << "  width: " << width_x << "  resolution: " << resolution);

  // set up tokens and get list of subscribers
  std::istringstream iss(topics);
  std::vector<std::string> tokens{std::istream_iterator<std::string>(iss), std::istream_iterator<std::string>()};

  for (const std::string &topic : tokens) {
    ROS_INFO_STREAM("Mapper subscribing to " << topic);
    subs.push_back(nh.subscribe<pcl::PointCloud<pcl::PointXYZ>>(topic, 1, boost::bind(frame_callback, _1, topic)));
  }

  // Timer for map decay
  if (decay_period > 0) {
    ros::Timer timer = nh.createTimer(ros::Duration(decay_period), decayMap);
  }

  published_map = std::unique_ptr<cv::Mat>(new cv::Mat(length_y, width_x, CV_8UC1));
  empty_map = std::unique_ptr<cv::Mat>(new cv::Mat(length_y, width_x, CV_8UC1));
  line_map = std::unique_ptr<cv::Mat>(new cv::Mat(length_y, width_x, CV_8UC1));
  barrel_map = std::unique_ptr<cv::Mat>(new cv::Mat(length_y, width_x, CV_8UC1));
  log_odds_map_lines = std::unique_ptr<Eigen::MatrixXf>(new Eigen::MatrixXf(length_y, width_x));
  log_odds_map_barrels = std::unique_ptr<Eigen::MatrixXf>(new Eigen::MatrixXf(length_y, width_x));

  map_pub = nh.advertise<igvc_msgs::map>("/map", 1);
  map_empty_pub = nh.advertise<igvc_msgs::map>("/empty_map", 1);

  if (debug) {
    debug_pub = nh.advertise<sensor_msgs::Image>("/map_debug", 1);
    debug_pcl_pub = nh.advertise<pcl::PointCloud<pcl::PointXYZRGB>>("/map_debug_pcl", 1);
    debug_empty_pcl_pub = nh.advertise<pcl::PointCloud<pcl::PointXYZRGB>>("/map_empty_debug_pcl", 1);
    debug_log_odds_pcl_pub = nh.advertise<pcl::PointCloud<pcl::PointXYZRGB>>("/map_log_odds_pcl_pub", 1);
  }

  ros::spin();
}
