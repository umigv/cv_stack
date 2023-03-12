#!/usr/bin/env python
import rospy
import numpy as np
import cv2

from sensor_msgs.msg import Image, PointCloud2

import rospy
from std_msgs.msg import String

import sys
from cv_bridge import CvBridge
from rospy.numpy_msg import numpy_msg
import message_filters

# camera params, should not change ever
FX = 708.4204711914062
FY = 708.4204711914062
CX = 545.5575561523438
CY = 333.53106689453125

def mess_with_depth_map(cv_image, depth_image):

    # mask = np.ones_like(cv_image) * 255

    # masked_image = cv2.bitwise_and(cv_image, mask)
    depth_to_point_cloud(depth_image)

def depth_to_point_cloud(img):
    # shape (621, 1104)
    data = []
    # iterate through each pixel :(
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            depth_val = img[x][y]
            if depth_val > 0:
                # Compute the x, y, and z coordinates of each point in the point cloud
                X = (x - CX) * depth_val / FX
                Y = (y - CY) * depth_val / FY
                Z = depth_val
                data.append((X, Y, Z))

    publish_point_cloud(np.array(data).astype(np.uint8).ravel().tolist())

def publish_point_cloud(data):
    cloud = PointCloud2()
    cloud.header.stamp = rospy.Time.now()
    cloud.header.frame_id = 'map'
    cloud.data = data

    topic = '/cv/laneMapping/point_cloud'
    pub = rospy.Publisher(topic, PointCloud2, queue_size=3)

    pub.publish(cloud)
    rospy.loginfo("Published point cloud")


def callback(data, depth_map):
    #rospy.loginfo("Converting perspective transformed img to edge detection image")
    bridge = CvBridge()
    timestamp = data.header.stamp
    cv_image = bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
    depth_image = bridge.imgmsg_to_cv2(depth_map, desired_encoding='passthrough')
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

    image_sub = message_filters.Subscriber("/zed/zed_node/left/image_rect_gray", Image, queue_size = 1, buff_size=2**24)
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