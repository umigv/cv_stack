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

    def potholes(self, img, threshold):
        '''
        params = cv2.SimpleBlobDetector_Params()

        # min_threshold = 50
        # max_threshold = 220
        # threshold_step = 10
        # min_repeatability = 2
        # min_dist_between_blobs = 10
        params.filterByArea = 1
        params.minArea = 25
        params.maxArea = 10000000

        params.filterByCircularity = 1
        params.minCircularity = 0.8
        params.maxCircularity = 2
        
        detector = cv2.SimpleBlobDetector_create(params)

        keypoints = detector.detect(img)

        im_with_keypoints = cv2.drawKeypoints(threshold, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        '''
        #print(len(img.shape))

        img = cv2.normalize(src=img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,30,
                            param1=300,param2=18,minRadius=0,maxRadius=30)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0,:]:
                # draw the outer circle
                cv2.circle(threshold,(i[0],i[1]),i[2],(255,255,255),2)
                # draw the center of the circle
                #cv2.circle(threshold,(i[0],i[1]),2,(0,0,255),3)
        return None


    def colorThreshold(self, img):

        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        # define range of white color in HSV
        # change it according to your need !
        lower_white = np.array([0,150,0], dtype=np.uint8)
        upper_white = np.array([255,255,255], dtype=np.uint8)
        # Threshold the HSV image to get only white colors
        mask = cv2.inRange(hls, lower_white, upper_white)
        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(hls, hls, mask= mask)

        return res


    def returnGaussianBlur(self, depth_image):

        # there's no hls to gray conversion, so have to go to bgr first
        # can also try extracting just the luminance column
        gaussianBlur = cv2.cvtColor(gaussianBlur, cv2.COLOR_HLS2BGR)
        gaussianBlur = cv2.cvtColor(gaussianBlur, cv2.COLOR_BGR2GRAY)

        masked_image = cv2.bitwise_and(depth_image, depth_image, mask=gaussianBlur)

        return masked_image

    def returnEDImage(self, depth_image):

        masked_image = cv2.bitwise_and(depth_image, depth_image, mask=edgeDetectedImage)

        return masked_image

    def returnThresholdedImage(self, depth_image):

        # there's no hls to gray conversion, so have to go to bgr first
        # can also try extracting just the luminance column
        thresholded = cv2.cvtColor(thresholded, cv2.COLOR_HLS2BGR)
        thresholded = cv2.cvtColor(thresholded, cv2.COLOR_BGR2GRAY)

        masked_image = cv2.bitwise_and(depth_image, depth_image, mask=thresholded)

        return masked_image

    def returnCroppedImage(self, depth_image):

        # there's no hls to gray conversion, so have to go to bgr first
        # can also try extracting just the luminance column
        cropped = cv2.cvtColor(cropped, cv2.COLOR_HLS2BGR)
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

        masked_image = cv2.bitwise_and(depth_image, depth_image, mask=cropped)

        return masked_image
    

    def __init__(self, image):

        # color threshold
        global thresholded
        thresholded = self.colorThreshold(image)

        # apply gaussian blur
        global gaussianBlur
        kernelSize = 5
        gaussianBlur = self.gaussian_blur(thresholded, kernelSize)

        grayImage = self.grayscale(gaussianBlur)
        
        # canny
        minThreshold = 150
        maxThreshold = 230
        global edgeDetectedImage
        edgeDetectedImage = self.cannyEdgeDetection(grayImage, minThreshold, maxThreshold)

    
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

    def region_of_interest(self, img, vertices):
        """
        Applies an image mask.

        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        """
        # decided to take this out for now since old definition of the region of
        # interest is not the same as our region of interest now,
        # feel free to add code here
        mask = np.zeros_like(image)
        match_mask_color = 255
        cv2.fillPoly(mask, np.int32([vertices]), match_mask_color)
        masked_image = cv2.bitwise_and(image, mask)
        return masked_image


def callback(image, depth_image, camera_info):
    #rospy.loginfo("Converting perspective transformed img to edge detection image")
    bridge = CvBridge()
    timestamp = image.header.stamp
    cv_image = bridge.imgmsg_to_cv2(image, desired_encoding='bgr8')
    depth_image = bridge.imgmsg_to_cv2(depth_image, desired_encoding='passthrough')
    ads = ADSDetection(cv_image)
    publish_transform(bridge, "/cv/laneMapping/left", ads.returnEDImage(depth_image), timestamp)
    """
    we are publishing this because for some reason the depth.launch
    file is looking for a topic with this exact name, even though
    I feel that it should be able to find /zed/zed_node/depth/camera_info
    This is a workaround, we are not editing the camera info at all
    """
    publish_camera_info("/cv/laneMapping/camera_info", camera_info, timestamp)

def publish_transform(bridge, topic, cv2_img, timestamp):
    transformed_ros = bridge.cv2_to_imgmsg(cv2_img)
    transformed_ros.header.stamp = timestamp
    transformed_ros.header.frame_id = "zed_left_camera_optical_frame"
    pub = rospy.Publisher(topic, Image, queue_size=1)
    pub.publish(transformed_ros)
    rospy.loginfo("Published transform")

def publish_camera_info(topic, camera_info, timestamp):
    camera_info.header.stamp = timestamp
    pub = rospy.Publisher(topic, CameraInfo, queue_size=1)
    pub.publish(camera_info)
    rospy.loginfo("Published camera info")

def listener():
    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('ads_detection', anonymous=True)
    global POTHOLES_ENABLED
    msg = "Initialized Cuda Detection: " + "No potholes" if not POTHOLES_ENABLED else "Potholes Enabled"
    rospy.loginfo(msg)

    image_sub = message_filters.Subscriber("/zed/zed_node/left/image_rect_gray", Image, queue_size = 1, buff_size=2**24)
    # image_sub = message_filters.Subscriber("/zed/zed_node/left/image_rect_color", Image)
    depth_map_sub = message_filters.Subscriber("/zed/zed_node/depth/depth_registered", Image, queue_size = 1, buff_size=2**24)
    
    camera_info_sub = message_filters.Subscriber("/zed/zed_node/depth/camera_info", CameraInfo, queue_size = 1, buff_size=2**24)
    ts = message_filters.TimeSynchronizer([image_sub, depth_map_sub, camera_info_sub], 10)
    ts.registerCallback(callback)

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
