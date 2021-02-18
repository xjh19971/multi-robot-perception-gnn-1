#include "flightros/flight_pilot.hpp"
#include <rosparam_shortcuts/rosparam_shortcuts.h>
#include "camera_pose.cpp"

/*
double find_norm(Vector<3> arr);
Matrix<3,3> matrix_dotProduct (Matrix<3,3> m, Matrix<3,3> n);
Matrix<3,3> rotation_matrix_from_vectors(Vector<3> o1, Vector<3> o2, Vector<3> B_r_BC);
Vector<3> initilization(cv::Mat image, Vector<3> c1); //initiliaze o1
double get_depth(cv::Mat image);
Vector<3> o1_position(Vector<3> c1, Vector<3> o1Direction, (Scalar) depth1);
Vector<3> find_o2(Matrix<3,3> R_BC, Vector<3> B_r_BC, Vector<3> o1, (Scalar) depth);
*/

namespace flightros {

FlightPilot::FlightPilot(const ros::NodeHandle &nh, const ros::NodeHandle &pnh)
  : nh_(nh),
    it_(nh),
    pnh_(pnh),
    scene_id_(UnityScene::WAREHOUSE),
    unity_ready_(false),
    unity_render_(false),
    receive_id_(0),
    main_loop_freq_(50.0),
    width_(720),
    height_(480),
    fov_(90),
    num_camera_(1)    {
  // load parameters
  if (!loadParams()) {
    ROS_WARN("[%s] Could not load all parameters.",
             pnh_.getNamespace().c_str());
  } else {
    ROS_INFO("[%s] Loaded all parameters.", pnh_.getNamespace().c_str());
  }

  // quad initialization
  quad_ptr_ = std::make_shared<Quadrotor>();

/* ----------------------------------------------------------------
  // add mono camera
  rgb_camera_ = std::make_shared<RGBCamera>();
  Vector<3> B_r_BC(0.0, 0.0, 0.3);
  Matrix<3, 3> R_BC = Quaternion(1.0, 0.0, 0.0, 0.0).toRotationMatrix();
  std::cout << R_BC << std::endl;
  rgb_camera_->setFOV(fov_);
  rgb_camera_->setWidth(width_);
  rgb_camera_->setHeight(height_);
  rgb_camera_->setRelPose(B_r_BC, R_BC);
  rgb_camera_->setPostProcesscing(std::vector<bool>{true, false, false});
  quad_ptr_->addRGBCamera(rgb_camera_);

  // initialize publisher call backs
  pub_RGBImage_ = it_.advertise("flight_pilot/RGBImage", 1);
  pub_DepthMap_ = it_.advertise("flight_pilot/DepthMap", 1);
  // pub_Segmentation_ = it_.advertise("flight_pilot/Segmentation", 1);
  // pub_OpticalFlow_ = it_.advertise("flight_pilot/OpticalFlow", 1);
---------------------------------------------------------------- */ 

  // add multiple_cameras
  for (size_t i = 0; i < num_camera_; ++i){
    rgb_cameras_.push_back(std::make_shared<RGBCamera>());
    Vector<3> B_r_BC ((Scalar)relpose_T_[i][0],(Scalar)relpose_T_[i][1],(Scalar)relpose_T_[i][2]); 
    Matrix<3, 3> R_BC = Quaternion((Scalar)relpose_R_[i][0], (Scalar)relpose_R_[i][1], (Scalar)relpose_R_[i][2], (Scalar)relpose_R_[i][3]).toRotationMatrix();
    std::cout << R_BC << std::endl;
    rgb_cameras_[i]->setFOV(fov_);
    rgb_cameras_[i]->setWidth(width_);
    rgb_cameras_[i]->setHeight(height_);
    rgb_cameras_[i]->setRelPose(B_r_BC, R_BC);
    rgb_cameras_[i]->setPostProcesscing(std::vector<bool>{true, false, false});
    quad_ptr_->addRGBCamera(rgb_cameras_[i]);
    pub_RGBImages_.push_back(it_.advertise("flight_pilot/"+camera_names_[i]+"/RGBImage", 1));
    pub_DepthMaps_.push_back(it_.advertise("flight_pilot/"+camera_names_[i]+"/DepthMap", 1));
  }
  // initialize publisher call backs
  
  ROS_INFO("[%s] Initilized all cameras.", pnh_.getNamespace().c_str());

  // initialization
  quad_state_.setZero();
  quad_ptr_->reset(quad_state_);


  // initialize subscriber call backs
  sub_state_est_ = nh_.subscribe("flight_pilot/state_estimate", 1,
                                 &FlightPilot::poseCallback, this);

  timer_main_loop_ = nh_.createTimer(ros::Rate(main_loop_freq_),
                                     &FlightPilot::mainLoopCallback, this);


  // wait until the gazebo and unity are loaded
  ros::Duration(5.0).sleep();

  // connect unity
  setUnity(unity_render_);
  connectUnity();
}

FlightPilot::~FlightPilot() {}

void FlightPilot::poseCallback(const nav_msgs::Odometry::ConstPtr &msg) {
  quad_state_.x[QS::POSX] = (Scalar)msg->pose.pose.position.x;
  quad_state_.x[QS::POSY] = (Scalar)msg->pose.pose.position.y;
  quad_state_.x[QS::POSZ] = (Scalar)msg->pose.pose.position.z;
  quad_state_.x[QS::ATTW] = (Scalar)msg->pose.pose.orientation.w;
  quad_state_.x[QS::ATTX] = (Scalar)msg->pose.pose.orientation.x;
  quad_state_.x[QS::ATTY] = (Scalar)msg->pose.pose.orientation.y;
  quad_state_.x[QS::ATTZ] = (Scalar)msg->pose.pose.orientation.z;
  //
  quad_ptr_->setState(quad_state_);

  if (unity_render_ && unity_ready_) {
    unity_bridge_ptr_->getRender(0);
    unity_bridge_ptr_->handleOutput();
  }
}

//------------Modification begins----------------
//calculate o1
//two inputs: 1. depth image 2.initial angle of camera1
Vector<3> o1;
o1 = Pose.initilization(cv:Mat image, Vector<3> ((Scalar)relpose_R_[0][1], (Scalar)relpose_R_[0][2], (Scalar)relpose_R_[0][3]));
Vector<3> o1Direction;
/*
vector<double> initial_v;
initial_v.push_back(0.0);
initial_v.push_back(1.0);
initial_v.push_back(0.0);
    
vector<double> o1;
o1 = initilization(cv:mat image, vector<double> initial_v);
*/

void FlightPilot::mainLoopCallback(const ros::TimerEvent &event) { //refresh at every frame, 30 fps
    FrameID frame_id = 1;
    unity_bridge_ptr_->getRender(frame_id);
    bool handle_output = unity_bridge_ptr_->handleOutput();
    cv::Mat image;
    sensor_msgs::ImagePtr msg;
    /* ------------------------
    // mono camera publish
    rgb_camera_->getRGBImage(image);
    msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg();
    pub_RGBImage_.publish(msg);
    rgb_camera_->getDepthMap(image);
    msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg();
    pub_DepthMap_.publish(msg);
    
    // rgb_camera_->getSegmentation(image);
    // msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg();
    // pub_Segmentation_.publish(msg);
    // rgb_camera_->getOpticalFlow(image);
    // msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg();
    // pub_OpticalFlow_.publish(msg);
    ------------------------ */


    //o1 position is returned
    // o1 is only called once during entire duration(at the first screenshot)
    
    //the functions below are called whenever you take a screen shot, except the first screenshot.
    //you take the 2nd screenshot here
    Vector<3> depth2;
    Vector<3> o2;
    image2 = Pose.getDepthMap(image);
    depth2 = Pose.get_depth(image2);
    depth_image_= cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::TYPE_32FC1)->image;
    Vector<3> B_r_BC_1 ((Scalar)relpose_T_[0][0],(Scalar)relpose_T_[0][1],(Scalar)relpose_T_[0][2]); 
    Matrix<3, 3> R_BC_1 = Quaternion((Scalar)relpose_R_[0][0], (Scalar)relpose_R_[0][1], (Scalar)relpose_R_[0][2], (Scalar)relpose_R_[0][3]).toRotationMatrix();
    //we now have the second depth image from the code above
    //calculate o2 below, so we can find rotation matrix for the rest of the cameras
    //camera 1's angle is fixed all the time
    Vector<3> o2 = Pose.find_o2(Matrix<3,3> R_BC, Vector<3> B_r_BC, Vector<3> o1, (Scalar) depth);

    //use o1 ,o2 and current camera_relpose T to calculate rotation matrix
    for(size_t i = 1; i < num_camera_; ++i){
      // calculating rotation matrix for camera2-6
      Matrix<3, 3> R_BC = Pose.rotation_matrix_from_vectors(Vector<3> o1, Vector<3> o2, Vector<3> ((Scalar)relpose_T_[i][0],(Scalar)relpose_T_[i][1],(Scalar)relpose_T_[i][2]));
      //Matrix<3,3> R_BC = R_BC.toRotationMatrix();
      
      rgb_cameras_.push_back(std::make_shared<RGBCamera>());
      Vector<3> B_r_BC ((Scalar)relpose_T_[i][0],(Scalar)relpose_T_[i][1],(Scalar)relpose_T_[i][2]); 
      Matrix<3, 3> R_BC = Quaternion((Scalar)relpose_R_[i][0], (Scalar)relpose_R_[i][1], (Scalar)relpose_R_[i][2], (Scalar)relpose_R_[i][3]).toRotationMatrix();
      std::cout << R_BC << std::endl;
      rgb_cameras_[i]->setFOV(fov_);
      rgb_cameras_[i]->setWidth(width_);
      rgb_cameras_[i]->setHeight(height_);
      rgb_cameras_[i]->setRelPose(B_r_BC, R_BC);
    }


    for (size_t i = 0; i < num_camera_; ++i){
      rgb_cameras_[i]->getRGBImage(image);
      pub_RGBImages_[i].publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg());
      rgb_cameras_[i]->getDepthMap(image);
      pub_DepthMaps_[i].publish(cv_bridge::CvImage(std_msgs::Header(), "32FC1", image).toImageMsg());
    }
    Vector<3> o1 = o2; //set current o2 as next o1 
}

//---------NOT modified below------------
bool FlightPilot::setUnity(const bool render) {
  unity_render_ = render;
  if (unity_render_ && unity_bridge_ptr_ == nullptr) {
    // create unity bridge
    unity_bridge_ptr_ = UnityBridge::getInstance();
    unity_bridge_ptr_->addQuadrotor(quad_ptr_);
    ROS_INFO("[%s] Unity Bridge is created.", pnh_.getNamespace().c_str());
  }
  return true;
}

bool FlightPilot::connectUnity() {
  if (!unity_render_ || unity_bridge_ptr_ == nullptr) return false;
  unity_ready_ = unity_bridge_ptr_->connectUnity(scene_id_);
  return unity_ready_;
}

bool FlightPilot::loadParams(void) {
  // load parameters
  quadrotor_common::getParam("main_loop_freq", main_loop_freq_, pnh_);
  quadrotor_common::getParam("unity_render", unity_render_, pnh_);
  std::size_t error= 0;
  error += !rosparam_shortcuts::get("flight_pilot", pnh_, "scene_id", scene_id_);
  error += !rosparam_shortcuts::get("flight_pilot", pnh_, "fov", fov_);
  error += !rosparam_shortcuts::get("flight_pilot", pnh_, "width", width_);
  error += !rosparam_shortcuts::get("flight_pilot", pnh_, "height", height_);
  error += !rosparam_shortcuts::get("flight_pilot", pnh_, "num_camera", num_camera_);
  error +=!rosparam_shortcuts::get("flight_pilot", pnh_, "camera_names", camera_names_);
  for (size_t i = 0; i < num_camera_; ++i)
  {
    std::vector<double> tvec;
    std::vector<double> rvec;
    error += !rosparam_shortcuts::get("flight_pilot", pnh_, camera_names_[i]+"/relpose_T", tvec);
    relpose_T_.push_back(tvec);
    error += !rosparam_shortcuts::get("flight_pilot", pnh_, camera_names_[i]+"/relpose_R", rvec);
    relpose_R_.push_back(rvec);
  }
  //quadrotor_common::getParam("RelPose_T", relpose_T_, pnh_);
  //quadrotor_common::getParam("RelPose_R", relpose_R_, pnh_);
  return true;
}

}  // namespace flightros
