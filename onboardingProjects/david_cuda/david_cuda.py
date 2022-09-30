#!/usr/bin/env python
#importing some useful packages
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import rospy
from sensor_msgs.msg import Image
import numpy as np
import cv2
import math
import sys
from cv_bridge import CvBridge
from rospy.numpy_msg import numpy_msg
# from rospy_tutorials.msg import Floats
import message_filters
import ros_numpy
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import MapMetaData
from geometry_msgs.msg import Pose
import cupy as cp

def main():
    image = cv2.imread('../LaneImage.jpg')

    gray = self.grayscale(image)

    gpu_frame = cv2.cuda_GpuMat()
    gpu_frame.upload(gray)

    # apply gaussian blur
    kernelSize = 5
    gaussianBlur = self.gaussian_blur(gpu_frame, kernelSize)



    # canny
    minThreshold = 20
    maxThreshold = 100
    global edgeDetectedImage
    edgeDetectedImage = self.cannyEdgeDetection(gaussianBlur, minThreshold, maxThreshold)
    edgeImage_cpu = edgeDetectedImage.download
    rospy.loginfo(type(edgeImage_cpu))

    #apply mask
    # lowerLeftPoint = [130, 540]
    # upperLeftPoint = [410, 350]
    # upperRightPoint = [570, 350]
    # lowerRightPoint = [915, 540]

    # pts = np.array([[lowerLeftPoint, upperLeftPoint, upperRightPoint,
    # lowerRightPoint]], dtype=np.int32)
    # masked_image = region_of_interest(edgeDetectedImage, pts)


    # hough lines
    rho = 1
    theta = np.pi/180
    # originally 30
    threshold = 40
    min_line_len = 1
    max_line_gap = 20
    global houged
    houged = self.hough_lines(edgeDetectedImage, rho, theta,
                    threshold, min_line_len, max_line_gap, image)
    bottomLeft, bottomRight, topLeft, topRight = dst_points
    bottomLeft = [bottomLeft[0]+5, bottomLeft[1]]
    bottomRight = [bottomRight[0]-5, bottomRight[1]]
    topLeft = [topLeft[0]+5, topLeft[1]]
    topRight = [topRight[0]-5, topRight[1]]
    # needed to flip bottomRight and bottomLeft for fillPoly function to work correctly
    dst_points = [bottomRight, bottomLeft, topLeft, topRight]
    #rospy.loginfo(np.asarray(dst_points))
    houged = self.region_of_interest(houged, np.asarray(dst_points))
    leftArea = np.array([[0,0], topLeft, bottomLeft, [0,houged.shape[0]]])
    rightArea = np.array([topRight, [houged.shape[1],0], [houged.shape[1],houged.shape[0]], bottomRight])
    global thresholded
    gpu_frame.upload(houged)
    self.potholes(edgeImage_cpu, houged)
    thresholded = self.threshold(image, houged)
    cv2.fillPoly(thresholded, np.int32([leftArea]), color=(100, 100, 100))
    cv2.fillPoly(thresholded, np.int32([rightArea]), color=(100, 100, 100))

    # if we have a way to automatically set the unknown areas based on the 
    # dst_points in convertToOccupancy, rather than filling in the cv2 image
    # with fillPoly then we can kind of skip two steps




def grayscale(self,img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def gaussian_blur(self,img, kernel_size):
    """Applies a Gaussian Noise kernel"""

    #cpu
    # return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    #gpu
    #get an error saying rowFilter_ != 0 in function 'SeparableLinearFilter'
    f = cv2.cuda.createGaussianFilter(cv2.CV_8UC1, cv2.CV_8UC1, (kernel_size, kernel_size), 0)
    return f.apply(img)

def cannyEdgeDetection(self, img, minThreshold, maxThreshold):
    #cpu
    # return cv2.Canny(img, minThreshold, maxThreshold)

    #gpu
    detector = cv2.cuda.createCannyEdgeDetector(minThreshold, maxThreshold)
    return detector.detect(img)

def hough_lines(self,img, rho, theta, threshold, min_line_len, max_line_gap, originalImage):
    #added originalImage so that we can access the shape of the image
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    #cpu

    # lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
    #         minLineLength=min_line_len, maxLineGap=max_line_gap)
    # # import pdb; pdb.set_trace()
    # line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    # # rospy.loginfo(type(lines))
    # # rospy.loginfo(lines)
    # if isinstance(lines, np.ndarray):
    #     # rospy.loginfo(lines)
    #     self.draw_lines(line_img, lines)
    # return line_img

    #gpu

    detector = cv2.cuda.createHoughSegmentDetector(rho, theta, min_line_len, max_line_gap, 500)
    line = detector.detect(img)
    lines = line.download()
    #on the cpu, don't know how optimize this
    line_img = np.zeros((originalImage.shape[0], originalImage.shape[1], 3), dtype=np.uint8)
    # rospy.loginfo(type(lines))
    # rospy.loginfo(lines)
    if isinstance(lines, np.ndarray):
        # rospy.loginfo(lines)
        self.draw_lines(line_img, lines)
    return line_img

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


def draw_lines(self,img, lines, color=[255, 255, 255], thickness=2):
    """
    This function draws `lines` with `color` and `thickness`.
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


if __name__ == '__main__':
  main()
