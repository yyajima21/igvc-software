// Subscribes to Point Cloud Data, updates the occupancy grid, then publishes the data.

#include <cv_bridge/cv_bridge.h>
#include <igvc_msgs/map.h>
#include <math.h>
#include <nav_msgs/Odometry.h>
#include <pcl/point_cloud.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <ros/publisher.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <signal.h>
#include <stdlib.h>
#include <tf/transform_datatypes.h>
#include <tf_conversions/tf_eigen.h>
#include <visualization_msgs/Marker.h>
#include <Eigen/Core>
#include <igvc_utils/NodeUtils.hpp>
#include <igvc_utils/RobotState.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include "octomapper.h"

class Mapper
{
public:
  Mapper();

private:
  void pc_callback(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &pc);
  void publish(const cv::Mat &map, uint64_t stamp);
  void setMessageMetadata(igvc_msgs::map &message, sensor_msgs::Image &image, uint64_t pcl_stamp);
  bool checkExistsStaticTransform(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &msg, const std::string &topic);
  bool getOdomTransform(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &msg);

  cv_bridge::CvImage m_img_bridge;

  ros::Publisher m_map_pub;                                  // Publishes map
  ros::Publisher m_debug_pub;                                // Debug version of above
  ros::Publisher m_debug_pcl_pub;                            // Publishes map as individual PCL points
  ros::Publisher m_ground_pub;                               // Publishes ground points
  ros::Publisher m_nonground_pub;                            // Publishes non ground points
  ros::Publisher m_sensor_pub;                               // Publishes lidar position
  std::unique_ptr<cv::Mat> m_published_map;                  // Matrix will be publishing
  std::map<std::string, tf::StampedTransform> m_transforms;  // Map of static transforms TODO: Refactor this
  std::unique_ptr<tf::TransformListener> m_tf_listener;      // TF Listener

  double m_resolution;
  double m_transform_max_wait_time;
  int m_start_x;   // start x (m)
  int m_start_y;   // start y (m)
  int m_length_x;  // length (m)
  int m_width_y;   // width (m)
  bool m_debug;
  double m_radius;  // Radius to filter lidar points // TODO: Refactor to a new node
  std::string m_lidar_topic;
  RobotState m_state;          // Odom -> Base_link
  RobotState m_odom_to_lidar;  // Odom -> Lidar

  std::unique_ptr<Octomapper> m_octomapper;
  pc_map_pair m_pc_map_pair;  // Struct storing both the octomap and the cv::Mat map
};

Mapper::Mapper() : m_tf_listener{ std::unique_ptr<tf::TransformListener>(new tf::TransformListener()) }
{
  ros::NodeHandle nh;
  ros::NodeHandle pNh("~");

  igvc::getParam(pNh, "octree/resolution", m_resolution);
  igvc::getParam(pNh, "map/length", m_length_x);
  igvc::getParam(pNh, "map/width", m_width_y);
  igvc::getParam(pNh, "map/start_x", m_start_x);
  igvc::getParam(pNh, "map/start_y", m_start_y);
  igvc::getParam(pNh, "node/debug", m_debug);
  igvc::getParam(pNh, "sensor_model/max_range", m_radius);
  igvc::getParam(pNh, "node/lidar_topic", m_lidar_topic);

  m_octomapper = std::unique_ptr<Octomapper>(new Octomapper(pNh));
  m_octomapper->create_octree(m_pc_map_pair);

  ros::Subscriber pcl_sub = nh.subscribe<pcl::PointCloud<pcl::PointXYZ>>(m_lidar_topic, 1, &Mapper::pc_callback, this);

  m_published_map = std::unique_ptr<cv::Mat>(new cv::Mat(m_length_x, m_width_y, CV_8UC1));

  m_map_pub = nh.advertise<igvc_msgs::map>("/map", 1);

  if (m_debug)
  {
    m_debug_pub = nh.advertise<sensor_msgs::Image>("/map_debug", 1);
    m_debug_pcl_pub = nh.advertise<pcl::PointCloud<pcl::PointXYZRGB>>("/map_debug_pcl", 1);
    m_ground_pub = nh.advertise<pcl::PointCloud<pcl::PointXYZ>>("/ground_pcl", 1);
    m_nonground_pub = nh.advertise<pcl::PointCloud<pcl::PointXYZ>>("/nonground_pcl", 1);
    m_sensor_pub = nh.advertise<visualization_msgs::Marker>("/sensor_pos", 1);
  }

  ros::spin();
}

/**
 * Updates <code>RobotState state</code> with the latest tf transform using the timestamp of the message passed in
 * @param[in] msg <code>pcl::PointCloud</code> message with the timestamp used for looking up the tf transform
 */
bool Mapper::getOdomTransform(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &msg)
{
  tf::StampedTransform transform;
  tf::StampedTransform transform2;
  ros::Time messageTimeStamp;
  pcl_conversions::fromPCL(msg->header.stamp, messageTimeStamp);
  try
  {
    if (m_tf_listener->waitForTransform("/odom", "/base_link", messageTimeStamp,
                                        ros::Duration(m_transform_max_wait_time)))
    {
      m_tf_listener->lookupTransform("/odom", "/base_link", messageTimeStamp, transform);
      m_state.setState(transform);
      m_tf_listener->lookupTransform("/odom", "/lidar", messageTimeStamp, transform2);
      m_odom_to_lidar.setState(transform2);
      return true;
    }
    else
    {
      ROS_DEBUG("Failed to get transform from /base_link to /odom in time, using newest transforms");
      m_tf_listener->lookupTransform("/odom", "/base_link", ros::Time(0), transform);
      m_state.setState(transform);
      m_tf_listener->lookupTransform("/odom", "/lidar", ros::Time(0), transform2);
      m_odom_to_lidar.setState(transform2);
      return true;
    }
  }
  catch (const tf::TransformException &ex)
  {
    ROS_ERROR("%s", ex.what());
    return false;
  }
}

// Populates igvc_msgs::map message with information from sensor_msgs::Image and the timestamp from pcl_stamp
/**
 * Populates <code>igvc_msgs::map message</code> with information from <code>sensor_msgs::Image</code> and the
 * timestamp from <code>pcl_stamp</code>
 * @param[out] message message to be filled out
 * @param[in] image image containing map data to be put into <code>message</code>
 * @param[in] pcl_stamp time stamp from the pcl to be used for <code>message</code>
 */
void Mapper::setMessageMetadata(igvc_msgs::map &message, sensor_msgs::Image &image, uint64_t pcl_stamp)
{
  pcl_conversions::fromPCL(pcl_stamp, image.header.stamp);
  pcl_conversions::fromPCL(pcl_stamp, message.header.stamp);
  message.header.frame_id = "/odom";
  message.image = image;
  message.length = m_length_x / m_resolution;
  message.width = m_width_y / m_resolution;
  message.resolution = m_resolution;
  message.orientation = m_state.yaw();
  message.x = std::round(m_state.x() / m_resolution) + m_start_x / m_resolution;
  message.y = std::round(m_state.y() / m_resolution) + m_start_y / m_resolution;
  message.x_initial = m_start_x / m_resolution;
  message.y_initial = m_start_y / m_resolution;
}

/**
 * Checks if transform from base_footprint to msg.header.frame_id exists
 * @param[in] msg
 * @param[in] topic Topic to check for
 */
bool Mapper::checkExistsStaticTransform(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &msg, const std::string &topic)
{
  if (m_transforms.find(topic) == m_transforms.end())
  {
    // Wait for transform between frame_id (ex. /scan/pointcloud) and base_footprint.
    ros::Time messageTimeStamp;
    pcl_conversions::fromPCL(msg->header.stamp, messageTimeStamp);
    ROS_INFO_STREAM("Getting transform for " << topic << " from " << msg->header.frame_id << " to /base_footprint \n");
    if (m_tf_listener->waitForTransform("/base_footprint", msg->header.frame_id, messageTimeStamp, ros::Duration(3.0)))
    {
      tf::StampedTransform transform;
      m_tf_listener->lookupTransform("/base_footprint", msg->header.frame_id, messageTimeStamp, transform);
      m_transforms.insert(std::pair<std::string, tf::StampedTransform>(topic, transform));
      ROS_INFO_STREAM("Found static transform from " << msg->header.frame_id << " to /base_footprint");
    }
    else
    {
      ROS_ERROR_STREAM("Failed to find transform using empty transform");
      return false;
    }
  }
  return true;
}

/**
 * Publishes the given map at the given stamp
 * @param[in] map map to be published
 * @param[in] stamp pcl stamp of the timestamp to be used
 */
void Mapper::publish(const cv::Mat &map, uint64_t stamp)
{
  igvc_msgs::map message;    // >> message to be sent
  sensor_msgs::Image image;  // >> image in the message
  m_img_bridge = cv_bridge::CvImage(message.header, sensor_msgs::image_encodings::MONO8, map);
  m_img_bridge.toImageMsg(image);  // from cv_bridge to sensor_msgs::Image

  setMessageMetadata(message, image, stamp);
  m_map_pub.publish(message);
  if (m_debug)
  {
    m_debug_pub.publish(image);
    // ROS_INFO_STREAM("\nThe robot is located at " << state);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr fromOcuGrid = boost::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
    for (int i = 0; i < m_length_x / m_resolution; i++)
    {
      for (int j = 0; j < m_width_y / m_resolution; j++)
      {
        pcl::PointXYZRGB p;
        uchar prob = map.at<uchar>(i, j);
        p = pcl::PointXYZRGB();
        p.x = (i * m_resolution) - (m_width_y / 2);
        p.y = (j * m_resolution) - (m_length_x / 2);
        if (prob > 127)
        {
          p.r = 0;
          p.g = static_cast<uint8_t>((prob - 127) * 2);
          p.b = 0;
          fromOcuGrid->points.push_back(p);
        }
        else if (prob < 127)
        {
          p = pcl::PointXYZRGB();
          p.r = 0;
          p.g = 0;
          p.b = static_cast<uint8_t>((127 - prob) * 2);
          fromOcuGrid->points.push_back(p);
        }
      }
    }
    fromOcuGrid->header.frame_id = "/odom";
    fromOcuGrid->header.stamp = stamp;
    m_debug_pcl_pub.publish(fromOcuGrid);
  }
}

/**
 * Callback for pointcloud. Filters the lidar scan, then inserts it into the octree.
 * @param[in] pc Lidar scan
 */
void Mapper::pc_callback(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &pc)
{
  // Pass through filter to only keep ones closest to us
  pcl::PointCloud<pcl::PointXYZ>::Ptr small(new pcl::PointCloud<pcl::PointXYZ>);
  float dist;
  for (int point_i = 0; point_i < pc->size(); ++point_i)
  {
    dist = pc->at(point_i).x * pc->at(point_i).x + pc->at(point_i).y * pc->at(point_i).y +
           pc->at(point_i).z * pc->at(point_i).z;
    if (dist <= m_radius * m_radius * m_radius)
    {
      small->push_back(pc->at(point_i));
    }
  }

  // make transformed clouds
  pcl::PointCloud<pcl::PointXYZ>::Ptr transformed =
      pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());

  // Check if static transform already exists for this topic.
  if (!checkExistsStaticTransform(pc, "/scan/pointcloud"))
  {
    ROS_ERROR("Sleeping 2 seconds then trying again...");
    ros::Duration(2).sleep();
    return;
  }

  // Lookup transform form Ros Localization for position
  if (!getOdomTransform(pc))
  {
    ROS_ERROR("Sleeping 2 seconds then trying again...");
    ros::Duration(2).sleep();
    return;
  }

  // Apply transformation from lidar to base_link aka robot pose
  pcl_ros::transformPointCloud(*small, *transformed, m_transforms.at(m_lidar_topic));
  pcl_ros::transformPointCloud(*transformed, *transformed, m_state.transform);

  visualization_msgs::Marker points;
  points.header.frame_id = "/odom";
  points.header.stamp = ros::Time::now();
  points.pose.position.x = m_odom_to_lidar.transform.getOrigin().getX();
  points.pose.position.y = m_odom_to_lidar.transform.getOrigin().getY();
  points.pose.position.z = m_odom_to_lidar.transform.getOrigin().getZ();

  points.action = visualization_msgs::Marker::ADD;
  points.id = 0;
  points.type = visualization_msgs::Marker::CUBE;
  points.scale.x = 0.05;
  points.scale.y = 0.05;
  points.scale.z = 0.05;
  points.color.g = 1.0f;
  points.color.a = 1.0;

  m_sensor_pub.publish(points);

  m_octomapper->insert_scan(m_odom_to_lidar.transform.getOrigin(), m_pc_map_pair, *transformed);

  // Publish map
  m_octomapper->get_updated_map(m_pc_map_pair);
  publish(*(m_pc_map_pair.map), pc->header.stamp);
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "mapper");
  Mapper mapper;
}
