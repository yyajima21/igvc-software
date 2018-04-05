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
#include <cv_bridge/cv_bridge.h>
#include <image_geometry/pinhole_camera_model.h>
#include <image_transport/image_transport.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <tf/transform_listener.h>


#include "trtengine.cpp"

image_transport::Publisher pub;

void img_callback(const cv::Mat msg_img, const sensor_msgs::ImageConstPtr& origMsg)
{
  int i = 1;
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "tensorrt");

  ros::NodeHandle nh;

  ros::NodeHandle pNh("~");

  image_transport::ImageTransport _it(nh);

  image_transport::Subscriber map_sub = _it.subscribe("/usb_cam_center/image_raw", 1, img_callback);

  pub = _it.advertise("/semantic_segmentation", 1);

  ros::spin();

  return 0;
}
