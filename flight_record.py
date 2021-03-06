import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
from cv_bridge import CvBridge, CvBridgeError
import os
import cv2
import numpy as np
from PIL import Image as Im
num_camera = 5
camera_names = ['DroneNN_main', 'DroneNP_main', 'DronePN_main', 'DronePP_main', 'DroneZZ_main']
save_dir = '/media/data/dataset/flightmare-npy'
modality = ['scene','depth','depth_encoded','pose']
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
for j in range(len(modality)):
    print(save_dir+'/'+modality[j])
    if not os.path.exists((save_dir+'/'+modality[j])):
        os.mkdir(save_dir+'/'+modality[j])

for i in range(num_camera):
    for j in range(len(modality)):
        if not os.path.exists(save_dir+'/'+modality[j]+'/'+camera_names[i]):
            os.mkdir(save_dir+'/'+modality[j]+'/'+camera_names[i])

bridge = CvBridge()
frame = 0

rgb=Image()
depth=Image()
p=Pose()
rate_value=100
def image_callback(msg):
    global rgb
    #print("Received a RGB image!")
    try:
        # Convert your ROS Image message to OpenCV2
        rgb = bridge.imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError,e:
        print(e)

def depth_callback(msg):
    global depth
    #print("Recieved a depth image!")
    try:
        # Convert your ROS Image message to OpenCV2
        depth = bridge.imgmsg_to_cv2(msg, "32FC1")
        #print(np.max(depth), ' ', np.min(depth))
    except CvBridgeError,e:
        print(e)

def pose_callback(msg):
    global p
    #print("Received a pose")
    p.position.x = msg.position.x
    p.position.y = msg.position.y
    p.position.z = msg.position.z
    p.orientation.x = msg.orientation.x
    p.orientation.y = msg.orientation.y
    p.orientation.z = msg.orientation.z
    p.orientation.w = msg.orientation.w

def save_msg(timestamp, frame):
    global rgb,depth,p
    print('[save msg]')
    for i in range(num_camera):
        key = raw_input("camera "+str(i))
        rgb_path = save_dir + "/scene/" + camera_names[i] + "/frame" + str(frame).zfill(6)+".png"
        cv2.imwrite(rgb_path, rgb)
        depth_path = save_dir + "/depth/" + camera_names[i] + "/frame" + str(frame).zfill(6)+".png"
        depth_encoded_path = save_dir + "/depth_encoded/" + camera_names[i] + "/frame" + str(frame).zfill(6)+".npy"
        with open(depth_encoded_path, 'wb') as f:
            np.save(f, depth)
        cv2.imwrite(depth_path, depth)
        pose_path = save_dir + "/pose/" + camera_names[i] + "/pose" + str(frame).zfill(6)+".txt"
        file = open(pose_path, "w+")
        L = [str(timestamp)+'\n',str(p.position.x)+'\n', str(p.position.y)+'\n', str(p.position.z)+'\n', str(p.orientation.x)+'\n', str(p.orientation.y)+'\n', str(p.orientation.z)+'\n', str(p.orientation.w)+'\n']
        file.writelines(L)
        file.close()


def main():
    global frame
    rospy.init_node('image_listener')
    # Define your image topic
    image_topic = "/hummingbird/flight_pilot/camera_1/RGBImage"
    depth_topic = "/hummingbird/flight_pilot/camera_1/DepthMap"
    pose_topic = "/hummingbird/ground_truth/pose"
    rospy.Subscriber(image_topic, Image, image_callback)
    rospy.Subscriber(depth_topic, Image, depth_callback)
    rospy.Subscriber(pose_topic, Pose, pose_callback)
    rate = rospy.Rate(rate_value)
    frame = int(raw_input('Input Starting frame: (default = 0)'))
    while not rospy.is_shutdown():
        key = raw_input('next?')
        timestamp = rospy.Time.now()
        save_msg(timestamp,frame)
        frame+=1


if __name__ == '__main__':
    main()
