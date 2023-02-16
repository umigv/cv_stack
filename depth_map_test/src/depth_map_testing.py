#!/usr/bin/env python
import rospy
import numpy as np
import cv2

from sensor_msgs.msg import Image

import rospy
from std_msgs.msg import String

import sys
from cv_bridge import CvBridge
from rospy.numpy_msg import numpy_msg
import message_filters


def mess_with_depth_map(cv_image, depth_image):
    rospy.loginfo(depth_image[0][0])

def callback(data, depth_map):
    #rospy.loginfo("Converting perspective transformed img to edge detection image")
    bridge = CvBridge()
    timestamp = data.header.stamp
    cv_image = bridge.imgmsg_to_cv2(data, desired_encoding='bgra8')
    depth_image = bridge.imgmsg_to_cv2(data, desired_encoding='bgra8')
    mess_with_depth_map(cv_image, depth_image)
    # publish_transform(bridge, "/cv/laneMapping/left", ads.returnCroppedImage(), timestamp)


def listener():
    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('depth_map_testing', anonymous=True)
    rospy.loginfo("initialized depth_map_testing")

    image_sub = message_filters.Subscriber("/zed/zed_node/left/image_rect_color", Image, queue_size = 1, buff_size=2**24)
    # image_sub = message_filters.Subscriber("/zed/zed_node/left/image_rect_color", Image)
    depth_map_sub = message_filters.Subscriber("/zed/zed_node/depth/depth_registered", Image, queue_size = 1, buff_size=2**24)
    
    ts = message_filters.TimeSynchronizer([image_sub, depth_map_sub], 10)
    ts.registerCallback(callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    try:
        listener()
    #capture the Interrupt signals
    except rospy.ROSInterruptException:
        pass