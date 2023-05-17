import rospy
import sys
from sensor_msgs.msg import Image, CameraInfo
import numpy as np
import cv2
import sys
from cv_bridge import CvBridge
import message_filters


def main():
    image = cv2.imread("./lines.png")

    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    iterations = 3
    
    # image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    image = dilate(image, kernel, iterations)
    #image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    # # canny
    # minThreshold = 150
    # maxThreshold = 230
    # image = cannyEdgeDetection(image, minThreshold, maxThreshold)

    #image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)


    # # hough lines
    # rho = 1
    # theta = np.pi/180
    # # originally 30
    # hough_threshold = 80
    # min_line_len = 1
    # max_line_gap = 20
    # global houged
    # image = hough_lines(image, rho, theta, hough_threshold, min_line_len, max_line_gap)

    cv2.imshow("window", image)
    cv2.waitKey(0)

def dilate(img, kernel, iterations):
    return cv2.dilate(img, kernel, iterations=iterations)

def cannyEdgeDetection(img, minThreshold, maxThreshold):
    return cv2.Canny(img, minThreshold, maxThreshold)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
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
        draw_lines(line_img, lines)
    return line_img

def draw_lines(img, lines, color=[255, 255, 255], thickness=2):
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

if __name__ == '__main__':
    main()
