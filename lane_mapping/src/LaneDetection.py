#!/usr/bin/env python
# file for lane detection without perspective transform
# importing some useful packages
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import rospy
import sys
from sensor_msgs.msg import Image, CameraInfo
import numpy as np
import cv2
# import cupy as cp
import math
import sys
from cv_bridge import CvBridge
from rospy.numpy_msg import numpy_msg
import message_filters
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import MapMetaData
from geometry_msgs.msg import Pose


# Rajiv Andrew and David


class ADSDetection:
    
    def __init__(self, image, depth_image):

        # color threshold
        minThreshold = 150
        maxThreshold = 255
        global thresholded
        thresholded = self.colorThreshold(image, minThreshold, maxThreshold)

        # apply gaussian blur
        global gaussianBlur
        kernel_size = 5
        gaussianBlur = self.gaussian_blur(thresholded, kernel_size)

        # grayscale
        grayImage = self.grayscale(gaussianBlur)

        # # canny
        # minThreshold = 150
        # maxThreshold = 230
        # global edgeDetectedImage
        # edgeDetectedImage = self.cannyEdgeDetection(grayImage, minThreshold, maxThreshold)

        global maskedImage
        maskedImage = cv2.bitwise_and(depth_image, depth_image, mask=grayImage)

        global dilatedImage
        kernel_size = 5
        iterations = 5
        dilatedImage = self.dilate(maskedImage, kernel_size, iterations)
        # dilatedImage = maskedImage

    def colorThreshold(self, img, minThreshold, maxThreshold):

        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        # define range of white color in HSV
        # change it according to your need !
        lower_white = np.array([0,minThreshold,0], dtype=np.uint8)
        upper_white = np.array([255,maxThreshold,255], dtype=np.uint8)
        # Threshold the HSV image to get only white colors
        mask = cv2.inRange(hls, lower_white, upper_white)
        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(hls, hls, mask= mask)

        return res
    
    def cannyEdgeDetection(self, img, minThreshold, maxThreshold):
        return cv2.Canny(img, minThreshold, maxThreshold)

    def grayscale(self, img):
        """Applies the Grayscale transform
        This will return an image with only one color channel
        but NOTE: to see the returned image as grayscale
        you should call plt.imshow(gray, cmap='gray')"""
        # since the image will be in hls, we convert to bgr then gray
        image = cv2.cvtColor(img, cv2.COLOR_HLS2BGR)
        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return image

    def gaussian_blur(self, img, kernel_size):
        """Applies a Gaussian Noise kernel"""
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    
    def dilate(self, img, kernel_size, iterations):
        return cv2.dilate(img, (kernel_size, kernel_size), iterations=iterations)

    def returnThresholdedImage(self):

        # there's no hls to gray conversion, so have to go to bgr first
        # can also try extracting just the luminance column
        thresholded = cv2.cvtColor(thresholded, cv2.COLOR_HLS2BGR)
        thresholded = cv2.cvtColor(thresholded, cv2.COLOR_BGR2GRAY)

        return thresholded

    def returnGaussianBlur(self):

        # there's no hls to gray conversion, so have to go to bgr first
        # can also try extracting just the luminance column
        gaussianBlur = cv2.cvtColor(gaussianBlur, cv2.COLOR_HLS2BGR)
        gaussianBlur = cv2.cvtColor(gaussianBlur, cv2.COLOR_BGR2GRAY)

        return gaussianBlur

    def returnEDImage(self):

        return edgeDetectedImage
    
    def returnDilatedImage(self):
        global dilatedImage

        return dilatedImage

camera_info_ros = CameraInfo()
transformed_ros = Image()

def callback(image, depth_image, camera_info):
    #rospy.loginfo("Converting perspective transformed img to edge detection image")
    bridge = CvBridge()
    # timestamp = image.header.stamp
    cv_image = bridge.imgmsg_to_cv2(image, desired_encoding='bgr8')
    depth_image = bridge.imgmsg_to_cv2(depth_image, desired_encoding='passthrough')
    # depth_image = cv2.dilate(depth_image, (5, 5), iterations=10)
    ads = ADSDetection(cv_image, depth_image)
    publish_transform(bridge, ads.returnDilatedImage())
    """
    we are publishing this because for some reason the depth.launch
    file is looking for a topic with this exact name, even though
    I feel that it should be able to find /zed/zed_node/depth/camera_info
    This is a workaround, we are not editing the camera info at all
    """
    publish_camera_info(camera_info)

def publish_transform(bridge, cv2_img):
    global transformed_ros
    transformed_ros = bridge.cv2_to_imgmsg(cv2_img)
    transformed_ros.header.frame_id = "zed_left_camera_optical_frame"


    # pub = rospy.Publisher(topic, Image, queue_size=1)
    # pub.publish(transformed_ros)
    # rospy.loginfo("Published transform")

def publish_camera_info(camera_info):
    global camera_info_ros
    camera_info_ros = camera_info


    # pub = rospy.Publisher(topic, CameraInfo, queue_size=1)
    # pub.publish(camera_info)
    # rospy.loginfo("Published camera info")


def listener():
    global camera_info_ros
    global transformed_ros
    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('ads_detection', anonymous=True)
    global POTHOLES_ENABLED
    msg = "Initialized Lane Detection: " + "No potholes" if not POTHOLES_ENABLED else "Potholes Enabled"
    rospy.loginfo(msg)

    image_sub = message_filters.Subscriber("/zed/zed_node/left/image_rect_gray", Image, queue_size = 1, buff_size=2**24)
    # image_sub = message_filters.Subscriber("/zed/zed_node/left/image_rect_color", Image)
    depth_map_sub = message_filters.Subscriber("/zed/zed_node/depth/depth_registered", Image, queue_size = 1, buff_size=2**24)

    camera_info_sub = message_filters.Subscriber("/zed/zed_node/depth/camera_info", CameraInfo, queue_size = 1, buff_size=2**24)
    ts = message_filters.TimeSynchronizer([image_sub, depth_map_sub, camera_info_sub], 10)
    ts.registerCallback(callback)

    rate = rospy.Rate(10)
    transform_pub = rospy.Publisher("/cv/laneMapping/left", Image, queue_size=1)
    camera_info_pub = rospy.Publisher("/cv/laneMapping/camera_info", CameraInfo, queue_size=1)

    log = rospy.get_param("log_info")

    while not rospy.is_shutdown():
        rate.sleep()
        time = rospy.Time.now()
        transformed_ros.header.stamp = time
        camera_info_ros.header.stamp = time
        transform_pub.publish(transformed_ros)
        camera_info_pub.publish(camera_info_ros)
        if log:
            rospy.loginfo("Published transform and camera info")
        

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


if __name__ == '__main__':
    try:
        argv = rospy.myargv(argv=sys.argv)
        global POTHOLES_ENABLED
        if (len(argv) == 2):
            POTHOLES_ENABLED = True
        else:
            POTHOLES_ENABLED = False
        listener()
    except rospy.ROSInterruptException:
        pass
