#!/usr/bin/env python
# importing some useful packages
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

def main():
    image = cv2.imread('LaneImage.jpg', cv2.IMREAD_GRAYSCALE)
    gpu_frame = cv2.cuda_GpuMat()
    gpu_frame.upload(image)
    # show(image)
    # apply gaussian blur
    kernelSize = 5
    gaussianBlur = gaussian_blur(gpu_frame, kernelSize)

    # canny
    minThreshold = 20
    maxThreshold = 100
    edgeDetectedImage = cannyEdgeDetection(gaussianBlur, minThreshold, maxThreshold)
    
    # hough lines
    rho = 1
    theta = np.pi/180
    # originally 30
    threshold = 40
    min_line_len = 1
    max_line_gap = 20
    houged = hough_lines(edgeDetectedImage, rho, theta,
                    threshold, min_line_len, max_line_gap, image)
    
    pulled_image = edgeDetectedImage.download()

    # show(pulled_image)

    # #apply mask
    # # lowerLeftPoint = [130, 540]
    # # upperLeftPoint = [410, 350]
    # # upperRightPoint = [570, 350]
    # # lowerRightPoint = [915, 540]

    # # pts = np.array([[lowerLeftPoint, upperLeftPoint, upperRightPoint,
    # # lowerRightPoint]], dtype=np.int32)
    # # masked_image = region_of_interest(edgeDetectedImage, pts)


    
    # bottomLeft, bottomRight, topLeft, topRight = dst_points
    # bottomLeft = [bottomLeft[0]+5, bottomLeft[1]]
    # bottomRight = [bottomRight[0]-5, bottomRight[1]]
    # topLeft = [topLeft[0]+5, topLeft[1]]
    # topRight = [topRight[0]-5, topRight[1]]
    # # needed to flip bottomRight and bottomLeft for fillPoly function to work correctly
    # dst_points = [bottomRight, bottomLeft, topLeft, topRight]
    # #rospy.loginfo(np.asarray(dst_points))
    # houged = region_of_interest(houged, np.asarray(dst_points))
    # leftArea = np.array([[0,0], topLeft, bottomLeft, [0,houged.shape[0]]])
    # rightArea = np.array([topRight, [houged.shape[1],0], [houged.shape[1],houged.shape[0]], bottomRight])
    # global thresholded
    # gpu_frame.upload(houged)

def show(img):
    """Shows given image for 2 seconds."""
    cv2.imshow("Image",img)
    cv2.waitKey(2000)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    f = cv2.cuda.createGaussianFilter(cv2.CV_8UC1, cv2.CV_8UC1, (kernel_size, kernel_size), 0)
    f.apply(img, img)
    return img

def cannyEdgeDetection(img, minThreshold, maxThreshold):

    detector = cv2.cuda.createCannyEdgeDetector(minThreshold, maxThreshold)
    return detector.detect(img)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap, originalImage):
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
    #     draw_lines(line_img, lines)
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
        draw_lines(line_img, lines)
    return line_img

def region_of_interest(img, vertices):
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


def draw_lines(img, lines, color=[255, 255, 255], thickness=2):
    """
    This function draws `lines` with `color` and `thickness`.
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


if __name__ == '__main__':
  main()
