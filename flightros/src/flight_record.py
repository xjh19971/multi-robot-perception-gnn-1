import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
from cv_bridge import CvBridge, CvBridgeError

import cv2
num_camera = 5
camera_names = ['DroneNN_main', 'DroneNP_main', 'DronePN_main', 'DronePP_main', 'DroneZZ_main']
save_dir = '/media/data/dataset/flightmare'
bridge = CvBridge()
frame = 0

rgb=Image()
depth=Image()
p=Pose()
rate_value=100
def image_callback(msg):
    global rgb
    print("Received a RGB image!")
    try:
        # Convert your ROS Image message to OpenCV2
        rgb = bridge.imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError, e:
        print(e)
    #else:
        # Save your OpenCV2 image as a jpeg 
    #    cv2.imwrite('camera_image.jpeg', rgb)
def depth_callback(msg):
    global depth
    print("Recieved a depth image!")
    try:
        # Convert your ROS Image message to OpenCV2
        depth = bridge.imgmsg_to_cv2(msg, "mono8")
    except CvBridgeError, e:
        print(e)
    #else:
    #    # Save your OpenCV2 image as a jpeg 
    #    cv2.imwrite('depth_image.jpeg', depth)

def pose_callback(msg):
    global p
    print("Received a pose")
    p.orientation.x = msg.position.x
    p.orientation.y = msg.position.y
    p.orientation.z = msg.position.z
    p.orientation.x = msg.orientation.x
    p.orientation.y = msg.orientation.y
    p.orientation.z = msg.orientation.z
    p.orientation.w = msg.orientation.w

def save_msg():
    global rgb,depth,p 
    for i in range(num_camera):

    
def main():
    rospy.init_node('image_listener')
    # Define your image topic
    image_topic = "/hummingbird/flight_pilot/camera_1/RGBImage"
    depth_topic = "/hummingbird/flight_pilot/camera_1/DepthMap"
    pose_topic = "/hummingbird/ground_truth/pose"
    rospy.Subscriber(image_topic, Image, image_callback)
    rospy.Subscriber(depth_topic, Image, depth_callback)
    rospy.Subscriber(pose_topic, Pose, pose_callback)
    # Spin until ctrl + c
    #rospy.spin()
    rate = rospy.Rate(rate_value)
    while not rospy.is_shutdown():
        #cv2.imwrite('camera_image.jpeg', rgb)
        #cv2.imwrite('depth_image.jpeg', depth)
        print(p)
        rate.sleep()

if __name__ == '__main__':
    main()