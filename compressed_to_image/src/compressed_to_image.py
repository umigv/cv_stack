#!/usr/bin/env python
# license removed for brevity
# from tkinter import Image
import rospy
from sensor_msgs.msg import Image, Compressed_Image
from cv_bridge import CvBridge
import numpy as np
import sys

import cv2


def publish_image(bridge, topic, compressed_img, timestamp):
    cv2_img = bridge.compressed_imgmsg_to_cv2(compressed_img, desired_encoding='bgra8')
    transformed_ros = bridge.cv2_to_imgmsg(cv2_img)
    transformed_ros.header.stamp = timestamp
    pub = rospy.Publisher(topic, Image, queue_size=10)
    pub.publish(transformed_ros)
    rospy.loginfo("Published Image")

def listener():
    global TOPIC_TO_DECOMPRESS
    rospy.init_node('decompressed', anonymous=True)
    rospy.loginfo("Converting Compressed Image to Image")
    rospy.loginfo(TOPIC_TO_DECOMPRESS)
    rospy.Subscriber(TOPIC_TO_DECOMPRESS, CompressedImage, callback, queue_size = 1, buff_size=2**24)

    publish_image(bridge, f"{TOPIC_TO_DECOMPRESS}/decompressed", image, timestamp)
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


def init(argv):
    global TOPIC_TO_DECOMPRESS
    if (len(argv) > 1):
        TOPIC_TO_DECOMPRESS=argv[1]
    else:
        TOPIC_TO_DECOMPRESS="/zed/zed_node/left/image_rect_color"

if __name__ == '__main__':
    init(sys.argv)
    try:
        listener()
    except rospy.ROSInterruptException:
        pass

