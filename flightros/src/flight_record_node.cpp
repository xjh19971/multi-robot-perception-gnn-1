#include <ros/ros.h>

#include "flightros/flight_record.hpp"

int main(int argc, char** argv) {
  ros::init(argc, argv, "flight_record");
  flightros::FlightRecord Record(ros::NodeHandle(), ros::NodeHandle("~"));

  // spin the ros
  ros::spin();
  
  return 0;
}