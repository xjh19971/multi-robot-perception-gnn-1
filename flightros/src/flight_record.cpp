#include "flightros/flight_record.hpp"
#include <rosparam_shortcuts/rosparam_shortcuts.h>

namespace flightros {

FlightRecord::FlightRecord(const ros::NodeHandle &nh, const ros::NodeHandle &pnh)
  : nh_(nh),
    it_(nh),
    pnh_(pnh),
    main_loop_freq_(10.0),
    width_(256),
    height_(256),
    fov_(90),
    num_camera_(1)    {
  // load parameters
  if (!loadParams()) {
    ROS_WARN("[%s] Could not load all parameters.",
             pnh_.getNamespace().c_str());
  } else {
    ROS_INFO("[%s] Loaded all parameters.", pnh_.getNamespace().c_str());
  }

  // initialize subscriber call backs
  sub_state_est_ = nh_.subscribe("flight_pilot/state_estimate", 1, &FlightRecord::poseCallback, this);
  sub_RGBImages_= it_.subscribe("flight_pilot/"+camera_names_[0]+"/RGBImage", 1, &FlightRecord::RGBImageCallback, this);
  sub_DepthMaps_= it_.subscribe("flight_pilot/"+camera_names_[0]+"/DepthMap", 1, &FlightRecord::DepthMapCallback, this);
  timer_main_loop_ = nh_.createTimer(ros::Rate(main_loop_freq_),
                                     &FlightRecord::mainLoopCallback, this);
}

FlightRecord::~FlightRecord() {}

void FlightRecord::RGBImageCallback(const sensor_msgs::ImageConstPtr& msg) {
  rgb_image_= cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8)->image;
}

void FlightRecord::DepthMapCallback(const sensor_msgs::ImageConstPtr& msg) {
  depth_image_= cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::TYPE_32FC1)->image;
}

void FlightRecord::poseCallback(const nav_msgs::Odometry::ConstPtr &msg) {
  quad_state_.x[QS::POSX] = (Scalar)msg->pose.pose.position.x;
  quad_state_.x[QS::POSY] = (Scalar)msg->pose.pose.position.y;
  quad_state_.x[QS::POSZ] = (Scalar)msg->pose.pose.position.z;
  quad_state_.x[QS::ATTW] = (Scalar)msg->pose.pose.orientation.w;
  quad_state_.x[QS::ATTX] = (Scalar)msg->pose.pose.orientation.x;
  quad_state_.x[QS::ATTY] = (Scalar)msg->pose.pose.orientation.y;
  quad_state_.x[QS::ATTZ] = (Scalar)msg->pose.pose.orientation.z;
}

void FlightRecord::mainLoopCallback(const ros::TimerEvent &event) {
    FrameID frame_id = 1;
    cv::imwrite("/media/data/dataset/flightmare/rgb.png",rgb_image_);
    cv::imwrite("/media/data/dataset/flightmare/depth.png", depth_image_);

}


bool FlightRecord::loadParams(void) {
  // load parameters
  quadrotor_common::getParam("main_loop_freq", main_loop_freq_, pnh_);
  std::size_t error= 0;
  error += !rosparam_shortcuts::get("flight_record", pnh_, "fov", fov_);
  error += !rosparam_shortcuts::get("flight_record", pnh_, "width", width_);
  error += !rosparam_shortcuts::get("flight_record", pnh_, "height", height_);
  error += !rosparam_shortcuts::get("flight_record", pnh_, "num_camera", num_camera_);
  error +=!rosparam_shortcuts::get("flight_record", pnh_, "camera_names", camera_names_);
  for (size_t i = 0; i < num_camera_; ++i)
  {
    std::vector<double> tvec;
    std::vector<double> rvec;
    error += !rosparam_shortcuts::get("flight_record", pnh_, camera_names_[i]+"/relpose_T", tvec);
    relpose_T_.push_back(tvec);
    error += !rosparam_shortcuts::get("flight_record", pnh_, camera_names_[i]+"/relpose_R", rvec);
    relpose_R_.push_back(rvec);
  }
  //quadrotor_common::getParam("RelPose_T", relpose_T_, pnh_);
  //quadrotor_common::getParam("RelPose_R", relpose_R_, pnh_);
  return true;
}

}  // namespace flightros
