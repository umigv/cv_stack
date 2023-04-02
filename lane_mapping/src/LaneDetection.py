#!/usr/bin/env python
# file for lane detection without perspective transform
# importing some useful packages
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import rospy
import sys
from sensor_msgs.msg import Image
import numpy as np
import cv2
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
        
    
    def threshold(self, img, edges):
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        # define range of white color in HSV
        # change it according to your need !
        lower_white = np.array([0,175,0], dtype=np.uint8)
        upper_white = np.array([255, 255, 255], dtype=np.uint8)
        # Threshold the HLS image to get only white colors
        mask = cv2.inRange(hls, lower_white, upper_white)
        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(edges, edges, mask= mask)

        #morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel) #Can try this to clean up edges
        return res


    def returnGaussianBlur(self):
        return gaussianBlur

    def returnEDImage(self):
        return edgeDetectedImage

    def returnHougedImage(self):
        return houged

    def returnThresholdedImage(self):
        return thresholded

    def __init__(self, image, depth_image):
        
        # apply gaussian blur
        global gaussianBlur
         # apply gaussian blur
        kernelSize = 5
        gaussianBlur = self.gaussian_blur(image, kernelSize)
        gaussianBlur = self.gaussian_blur(gaussianBlur, kernelSize)
        # gaussianBlur = self.gaussian_blur(gaussianBlur, kernelSize)
        # gaussianBlur = self.gaussian_blur(gaussianBlur, kernelSize)
        # global thresholded
        # thresholded = self.threshold(gaussianBlur, gaussianBlur)

        gaussianBlur = cv2.convertScaleAbs(gaussianBlur, alpha=1, beta=0)
        global thresholded
        thresholded = self.threshold(gaussianBlur, gaussianBlur)

        # canny
        minThreshold = 100
        maxThreshold = 130
        global edgeDetectedImage
        edgeDetectedImage = thresholded
        # edgeDetectedImage = cv2.Canny(thresholded, minThreshold, maxThreshold)
        return
        rho = 1
        theta = np.pi/180
        # originally 30
        hough_threshold = 80
        min_line_len = 1
        max_line_gap = 20
        global houged
        houged = self.hough_lines(edgeDetectedImage, rho, theta,
                        hough_threshold, min_line_len, max_line_gap)
        # houged = region_of_interest(houged, np.asarray(dst_points))

        global POTHOLES_ENABLED
        if POTHOLES_ENABLED:
            self.potholes(edgeDetectedImage, houged)

        # self.convertToOccupancy(thresholded)
        # thresholded should be final occupancy grid

        

    def grayscale(self,img):
        """Applies the Grayscale transform
        This will return an image with only one color channel
        but NOTE: to see the returned image as grayscale
        you should call plt.imshow(gray, cmap='gray')"""
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def gaussian_blur(self,img, kernel_size):
        """Applies a Gaussian Noise kernel"""
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


    def region_of_interest(self,img, vertices):
        """
        Applies an image mask.

        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        """
        #defining a blank mask to start with
        mask = np.zeros_like(img)

        #defining a 3 channel or 1 channel color to fill the mask with
        #depending on the input image
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        #filling pixels inside the polygon defined by "vertices" with the fill color
        cv2.fillPoly(mask, np.int32([vertices]), ignore_mask_color)

        #returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image


    def hough_lines(self,img, rho, theta, threshold, min_line_len, max_line_gap):
        """
        `img` should be the output of a Canny transform.

        Returns an image with hough lines drawn.
        """
        lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
                minLineLength=min_line_len, maxLineGap=max_line_gap)
        # import pdb; pdb.set_trace()
        line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        # rospy.loginfo(type(lines))
        # rospy.loginfo(lines)
        if isinstance(lines, np.ndarray) and len(lines) > 1:
            # rospy.loginfo(lines)
            self.draw_lines(line_img, lines)
        return line_img



    def draw_lines(self,img, lines, color=[255, 255, 255], thickness=2):
        """
        This function draws `lines` with `color` and `thickness`.
        """
        lines = np.squeeze(lines)
        distances = np.linalg.norm(lines[:, 0:2] - lines[:, 2:], axis=1, keepdims=True)
        MAX_COLOR = 99
        MIN_COLOR = 0
        m = (MAX_COLOR - MIN_COLOR)/(np.max(distances) - np.min(distances)) # Resize distances between MIN and MAX COLOR
        color_range = m * distances
        THRESHOLD = 50 # Get rid of lines smaller than threshold

        # cv2.line is slow because of the for loop, but can be used to show various colors.
        # This can help if we want a continous probability of lines based on distance
        color_range[color_range < THRESHOLD] = 0
        for ((x1,y1,x2,y2), col) in zip(lines, color_range):
            cv2.line(img, (x1, y1), (x2, y2), [0, 0, 255], thickness)

        # cv2.polylines is faster but can draw only one color. Thus, the only type of filtering is binary with the threshold
        # filtered_lines = lines[(color_range > THRESHOLD).ravel(), :]
        # cv2.polylines(img, filtered_lines.reshape((-1, 2, 2)), False, 255, thickness) # Faster but can't draw multiple colors. Thus, no thresholding


    def convertToOccupancy(self, img):
        # function to take in the final lane detected image and convert into an occupancy grid
        height, width = img.shape[:2]
        resolution = 0.05 
        global grid
        grid = OccupancyGrid()
        m = MapMetaData()
        m.resolution = resolution
        m.width = width
        m.height = height
        pos = np.array([-width * resolution / 2, -height * resolution / 2, 0])
        m.origin = Pose()
        m.origin.position.x, m.origin.position.y = pos[:2]
        grid.info = m
        grid.data = self.convertImgToOgrid(img, height, width)


    def convertImgToOgrid(self, img, height, width):
        #function to take a cv2 image and return an int8[] array

        img = img[:, :, 0].astype(np.int16)
        img[img == 100] = -1
        # img[img > 0] = 100

        return img.astype(np.int8).ravel().tolist() # for ros msg


def callback(data, depth_map):
    #rospy.loginfo("Converting perspective transformed img to edge detection image")
    bridge = CvBridge()
    timestamp = data.header.stamp
    cv_image = bridge.imgmsg_to_cv2(data, desired_encoding='bgra8')
    depth_image = bridge.imgmsg_to_cv2(depth_map, desired_encoding='passthrough')
    ads = ADSDetection(cv_image, depth_image)

    publish_transform(bridge, "/cv/laneMapping/left", ads.returnEDImage(), timestamp)


def publish_transform(bridge, topic, cv2_img, timestamp):
    transformed_ros = bridge.cv2_to_imgmsg(cv2_img)
    transformed_ros.header.stamp = timestamp
    pub = rospy.Publisher(topic, Image, queue_size=1)
    pub.publish(transformed_ros)
    rospy.loginfo("Published transform")

def publish_ogrid(topic, ogrid_msg, timestamp):
    ogrid_msg.header.stamp = timestamp
    pub = rospy.Publisher(topic, OccupancyGrid, queue_size=1)
    #rospy.loginfo(ogrid_msg.data)
    pub.publish(ogrid_msg)
    rospy.loginfo("Published ogrid")

def listener():
    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('lane_detection', anonymous=True)
    global POTHOLES_ENABLED
    msg = "Initialized Lane Detection: " + "No potholes" if not POTHOLES_ENABLED else "Potholes Enabled"
    rospy.loginfo(msg)

    image_sub = message_filters.Subscriber("/zed/zed_node/left/image_rect_color", Image, queue_size = 1, buff_size=2**24)
    depth_map_sub = message_filters.Subscriber("/zed/zed_node/depth/depth_registered", Image, queue_size = 1, buff_size=2**24)

    ts = message_filters.TimeSynchronizer([image_sub, depth_map_sub], 10)
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
