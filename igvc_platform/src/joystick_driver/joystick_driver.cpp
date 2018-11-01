//TODO: Write general explanation of this node
#include <igvc_msgs/velocity_pair.h>
#include <ros/publisher.h>
#include <ros/ros.h>
#include <ros/subscriber.h>
#include <sensor_msgs/Joy.h>

ros::Publisher cmd_pub;

ros::NodeHandle* nhp;

//TODO: Write general documentation for method
//TODO: Refactor to use dynamic reconfigure instead of hacky parameter reading
void joyCallback(const sensor_msgs::Joy::ConstPtr& msg)
{
  double absoluteMaxVel, maxVel, maxVelIncr;
  //TODO: Move params to main or use dynamic reconfigure
  //TODO: Change absoluteMaxVel to hardcoded safety value
  nhp->param(std::string("absoluteMaxVel"), absoluteMaxVel, 1.0);
  nhp->param(std::string("maxVel"), maxVel, 1.6);
  //TODO: Rename maxVelIncr to deltaVel
  nhp->param(std::string("maxVelIncr"), maxVelIncr, 0.1);

  //TODO: Document design logic to explain Incr
  if (msg->buttons[1])
    maxVel -= maxVelIncr;
  else if (msg->buttons[3])
    maxVel += maxVelIncr;
  //TODO: Add warning if you are exceeding absoluteMaxVel
  maxVel = std::min(maxVel, absoluteMaxVel);
  maxVel = std::max(maxVel, 0.0);

  nhp->setParam("maxVel", maxVel);

  int leftJoyAxis, rightJoyAxis;
  bool leftInverted, rightInverted;
  //TODO: Move these params to main/dynamic reconfigure
  nhp->param(std::string("leftAxis"), leftJoyAxis, 1);
  nhp->param(std::string("rightAxis"), rightJoyAxis, 4);
  nhp->param(std::string("leftInverted"), leftInverted, false);
  nhp->param(std::string("rightInverted"), rightInverted, false);

  igvc_msgs::velocity_pair cmd;
  cmd.left_velocity = msg->axes[leftJoyAxis] * maxVel * (leftInverted ? -1.0 : 1.0);
  cmd.right_velocity = msg->axes[rightJoyAxis] * maxVel * (rightInverted ? -1.0 : 1.0);

  cmd_pub.publish(cmd);
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "joystick_driver");
  ros::NodeHandle nh;
  nhp = new ros::NodeHandle("~");

  cmd_pub = nh.advertise<igvc_msgs::velocity_pair>("/motors", 1);

  ros::Subscriber joy_sub = nh.subscribe("/joy", 1, joyCallback);

  ros::spin();

  delete nhp;

  return 0;
}
