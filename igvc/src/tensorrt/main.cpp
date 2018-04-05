#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <igvc_msgs/action_path.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <pcl/octree/octree_search.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <ros/publisher.h>
#include <ros/ros.h>
#include <ros/subscriber.h>
#include <tf/transform_datatypes.h>
#include <algorithm>
#include <mutex>
#include ""

ros::Publisher disp_path_pub;

void img_callback(const sensor_msgs::ImageConstPtr& msg, const sensor_msgs::CameraInfoConstPtr& cam_info)
{
  cv_bridge::CvImagePtr cv_copy;
  cv_copy = cv_bridge::toCvCopy(msg, "");
  cv::Mat img = cv_copy->image;

  img = ResizeCameraImage(img, 640, 360);
  sensor_msgs::CameraInfo cam_info_rsz = ResizeCameraInfo(cam_info, 640, 360);

  cam.fromCameraInfo(cam_info_rsz);
  img_callback(img, msg);
}

void LineDetector::img_callback(const cv::Mat msg_img, const sensor_msgs::ImageConstPtr& origMsg)
{
  src_img = msg_img;
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "tensorrt");

  ros::NodeHandle nh;

  ros::NodeHandle pNh("~");

  ros::Subscriber map_sub = nh.subscribe("/usb_cam_center/image_raw", 1, map_callback);

  act_path_pub = nh.advertise<igvc_msgs::action_path>("/semantic_segmentation", 1);

  ros::spin();

  return 0;
}
