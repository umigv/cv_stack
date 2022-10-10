#!/usr/bin/env python
# license removed for brevity
# from tkinter import Image
import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import numpy as np
import sys

import cv2


def callback(data):
    rospy.loginfo("Visualizing frame: " + TOPIC_TO_VISUALIZE)
    bridge = CvBridge()
    cv_image = bridge.compressed_imgmsg_to_cv2(data)
    global TOPIC_TO_VISUALIZE
    cv2.imshow(TOPIC_TO_VISUALIZE, cv_image)
    cv2.waitKey(1)


def listener():
    global TOPIC_TO_VISUALIZE
    rospy.init_node('visualizer', anonymous=True)
    rospy.loginfo("Visualizing Topic: ")
    rospy.loginfo(TOPIC_TO_VISUALIZE)
    rospy.Subscriber(TOPIC_TO_VISUALIZE, CompressedImage, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


def init(argv):
    global TOPIC_TO_VISUALIZE
    if (len(argv) > 1):
        TOPIC_TO_VISUALIZE=argv[1]
    else:
        TOPIC_TO_VISUALIZE="/zed/zed_node/left/image_rect_color"

if __name__ == '__main__':
    init(sys.argv)
    try:
        listener()
    except rospy.ROSInterruptException:
        pass

