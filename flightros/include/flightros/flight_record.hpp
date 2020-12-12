
#pragma once

#include <memory>

// ros
#include <nav_msgs/Odometry.h>
#include <ros/ros.h>

// rpg quadrotor
#include <autopilot/autopilot_helper.h>
#include <autopilot/autopilot_states.h>
#include <quadrotor_common/parameter_helper.h>
#include <quadrotor_msgs/AutopilotFeedback.h>

// flightlib
#include "flightlib/bridges/unity_bridge.hpp"
#include "flightlib/common/quad_state.hpp"
#include "flightlib/common/types.hpp"
#include "flightlib/objects/quadrotor.hpp"
#include "flightlib/sensors/rgb_camera.hpp"

//image
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
using namespace flightlib;

namespace flightros {

class FlightRecord {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  FlightRecord(const ros::NodeHandle& nh, const ros::NodeHandle& pnh);
  ~FlightRecord();

  // callbacks
  void mainLoopCallback(const ros::TimerEvent& event);
  void poseCallback(const nav_msgs::Odometry::ConstPtr& msg);
  void RGBImageCallback(const sensor_msgs::ImageConstPtr& msg);
  void DepthMapCallback(const sensor_msgs::ImageConstPtr& msg);
  bool loadParams(void);

 private:
  // ros nodes
  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;

  image_transport::ImageTransport it_;


  cv::Mat rgb_image_,depth_image_;
  // subscriber
  ros::Subscriber sub_state_est_;
  image_transport::Subscriber sub_RGBImages_;
  image_transport::Subscriber sub_DepthMaps_;
  // main loop timer
  ros::Timer timer_main_loop_;

  QuadState quad_state_;

  size_t num_camera_{1};
  std::vector< std::string> camera_names_;
  int fov_;
  int width_;
  int height_;
  std::vector< std::vector<double>> relpose_T_;
  std::vector< std::vector<double>> relpose_R_;
  // auxiliary variables
  Scalar main_loop_freq_{50.0};
};
}  // namespace flightros
