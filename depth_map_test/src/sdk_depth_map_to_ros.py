#!/usr/bin/env python
import rospy
import numpy as np
import cv2

from sensor_msgs.msg import Image
import pyzed.sl as sl
from cv_bridge import CvBridge

#function to publish messages at the rate of 2 messages per second
def messagePublisher():
    # set up Zed first
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE  # Use PERFORMANCE depth mode
    init_params.coordinate_units = sl.UNIT.FOOT  # Use meter units (for depth measurements)
    init_params.camera_resolution = sl.RESOLUTION.HD720

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        rospy.loginfo("unsuccessful zed opening")
        exit(1)
    else:
        rospy.loginfo("successful zed opening")


    # Create and set RuntimeParameters after opening the camera
    runtime_parameters = sl.RuntimeParameters()
    runtime_parameters.sensing_mode = sl.SENSING_MODE.STANDARD  # Use STANDARD sensing mode
    # Setting the depth confidence parameters
    runtime_parameters.confidence_threshold = 100
    runtime_parameters.textureness_confidence_threshold = 100

    depth = sl.Mat()

    # mirror_ref = sl.Transform()
    # mirror_ref.set_translation(sl.Translation(2.75,4.0,0))
    # tr_np = mirror_ref.m

    # Now set up ROS
    #define a topic to which the messages will be published
    message_publisher = rospy.Publisher('/cv/sdk_depth_map_test', Image, queue_size=1)
    #initialize the Publisher node. 
    rospy.init_node('sdk_depth_map_test', anonymous=True)
    msg = "Initialized SDK Depth map test"
    rospy.loginfo(msg)
    
    hz = 15
    rate = rospy.Rate(hz)
    bridge = CvBridge()
    
    # if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
    #     zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
    #     # zed.retrieve_image(depth, sl.VIEW.DEPTH)
    #     depth_map_arr = depth.get_data()
        
    #     np.savetxt('depth_map.csv', depth_map_arr, delimiter=',')
    #     rospy.loginfo("depth_map.csv saved")
    # return

    while not rospy.is_shutdown():

        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
            # zed.retrieve_image(depth, sl.VIEW.DEPTH)
            timestamp = rospy.get_rostime()

            depth_map_arr = depth.get_data()
            
            message = bridge.cv2_to_imgmsg(depth_map_arr)
            message.header.stamp = timestamp
            # rospy.loginfo(depth_map_arr[0])
            # message = Image()
            # message.width = depth_map_arr.shape[0]
            # message.height = depth_map_arr.shape[1]
            # print(type(depth_map_arr[0][0]))
            # message.data = depth_map_arr.astype(np.int8).tolist()
            # display the message on the terminal
            # publish the message to the topic
            message_publisher.publish(message)
            rospy.loginfo("published depth image")
        # rate.sleep() will wait enough until the node publishes the message to the topic
        rate.sleep()
if __name__ == '__main__':
    try:
        messagePublisher()
    #capture the Interrupt signals
    except rospy.ROSInterruptException:
        pass