#!/usr/bin/env python
import rospy
import numpy as np
import cv2

from sensor_msgs.msg import PointCloud2
import pyzed.sl as sl



#import the rospy package and the String message type
import rospy
from std_msgs.msg import String
#function to publish messages at the rate of 2 messages per second
def messagePublisher():
    # set up Zed first
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE  # Use PERFORMANCE depth mode
    init_params.coordinate_units = sl.UNIT.METER  # Use meter units (for depth measurements)
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

    image = sl.Mat()
    depth = sl.Mat()
    point_cloud = sl.Mat()

    mirror_ref = sl.Transform()
    mirror_ref.set_translation(sl.Translation(2.75,4.0,0))
    tr_np = mirror_ref.m
    
    # Now set up ROS
    #define a topic to which the messages will be published
    message_publisher = rospy.Publisher('/cv/point_cloud_test', PointCloud2, queue_size=1)
    #initialize the Publisher node. 
    rospy.init_node('point_cloud_test', anonymous=True)
    msg = "Initialized PointCloud2 test"
    rospy.loginfo(msg)

    while not rospy.is_shutdown():
        hz = 15
        rate = rospy.Rate(hz)
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
            # rospy.loginfo(point_cloud)
            point_cloud_np = point_cloud.get_data()
            point_cloud_np.dot(tr_np)
            message = point_cloud_np
            #display the message on the terminal
            #publish the message to the topic
            message_publisher.publish(message)
        # rate.sleep() will wait enough until the node publishes the message to the topic
        rate.sleep()
if __name__ == '__main__':
    try:
        messagePublisher()
    #capture the Interrupt signals
    except rospy.ROSInterruptException:
        pass