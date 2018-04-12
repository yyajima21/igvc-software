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
image_geometry::PinholeCameraModel cam;

cv::Mat ResizeCameraImage(cv::Mat oldImg, int width, int height)
{
  // Resize image
  cv::Mat retVal;
  cv::resize(oldImg, retVal, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
  return retVal;
}

sensor_msgs::CameraInfo ResizeCameraInfo(sensor_msgs::CameraInfoConstPtr oldCamInfo, int width, int height)
{
  // Change camera info
  boost::array<double, 9ul> newK = oldCamInfo->K;
  newK[0] *= (double)width / (double)oldCamInfo->width;
  newK[2] *= (double)width / (double)oldCamInfo->width;
  newK[4] *= (double)height / (double)oldCamInfo->height;
  newK[5] *= (double)height / (double)oldCamInfo->height;
  boost::array<double, 12ul> newP = oldCamInfo->P;
  newP[0] *= (double)width / (double)oldCamInfo->width;
  newP[2] *= (double)width / (double)oldCamInfo->width;
  newP[3] = 0;
  newP[5] *= (double)height / (double)oldCamInfo->height;
  newP[6] *= (double)height / (double)oldCamInfo->height;
  newP[7] = 0;

  // Update newCamInfo object
  sensor_msgs::CameraInfo cam_info_rsz = *oldCamInfo;
  cam_info_rsz.K = newK;
  cam_info_rsz.P = newP;
  cam_info_rsz.width = width;
  cam_info_rsz.height = height;
  return cam_info_rsz;
}

void img_callback(const cv::Mat msg_img, const sensor_msgs::ImageConstPtr& origMsg)
{
  cv_bridge::CvImagePtr newPtr = cv_bridge::toCvCopy(origMsg, "");
  newPtr->image = executeEngine(msg_img);
  pub.publish(newPtr->toImageMsg());
}

void info_img_callback(const sensor_msgs::ImageConstPtr& msg, const sensor_msgs::CameraInfoConstPtr& cam_info)
{
  cv_bridge::CvImagePtr cv_copy;
  cv_copy = cv_bridge::toCvCopy(msg, "");
  cv::Mat img = cv_copy->image;

  img = ResizeCameraImage(img, 640, 360);
  sensor_msgs::CameraInfo cam_info_rsz = ResizeCameraInfo(cam_info, 640, 360);

  cam.fromCameraInfo(cam_info_rsz);
  img_callback(img, msg);
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "tensorrt");

  ros::NodeHandle nh;

  ros::NodeHandle pNh("~");

  image_transport::ImageTransport _it(nh);

  loadEngine();

  image_transport::CameraSubscriber map_sub = _it.subscribeCamera("/usb_cam_center/image_raw", 1, info_img_callback);

  pub = _it.advertise("/semantic_segmentation", 1);

  ros::spin();

  return 0;
}
