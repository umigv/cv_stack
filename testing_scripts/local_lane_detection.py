#!/usr/bin/env python
#importing some useful packages
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import sys
import numpy as np
import cv2
    
# threshold using color from original image, and combine that with the houged image
def threshold(img, edges):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    # define range of white color in HSV
    # change it according to your need !
    lower_white = np.array([0,240,0], dtype=np.uint8)
    upper_white = np.array([255, 255, 255], dtype=np.uint8)
    # Threshold the HSV image to get only white colors
    mask = cv2.inRange(hsv, lower_white, upper_white)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(edges, edges, mask= mask)

    #morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel) #Can try this to clean up edges
    return res

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 200)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # decided to take this out for now since old definition of the region of
    # interest is not the same as our region of interest now,
    # feel free to add code here


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

    # cv2.polylines is faster but can draw only one color. Thus, the only type of filtering is binary with the threshold
    # filtered_lines = lines[(color_range > THRESHOLD).ravel(), :]
    # cv2.polylines(img, filtered_lines.reshape((-1, 2, 2)), False, 255, thickness) # Faster but can't draw multiple colors. Thus, no thresholding


def detect_lanes(image):

    # apply gaussian blur
    kernelSize = 5
    gaussianBlur = gaussian_blur(image, kernelSize)
    gaussianBlur = gaussian_blur(gaussianBlur, kernelSize)
    gaussianBlur = gaussian_blur(gaussianBlur, kernelSize)
    gaussianBlur = gaussian_blur(gaussianBlur, kernelSize)

    global thresholded
    thresholded = threshold(gaussianBlur, gaussianBlur)

    # canny
    minThreshold = 100
    maxThreshold = 130
    global edgeDetectedImage
    edgeDetectedImage = cv2.Canny(thresholded, minThreshold, maxThreshold)

    # hough lines
    rho = 1
    theta = np.pi/180
    # originally 30
    hough_threshold = 80
    min_line_len = 1
    max_line_gap = 20
    global houged
    houged = hough_lines(edgeDetectedImage, rho, theta,
                    hough_threshold, min_line_len, max_line_gap)
    # houged = region_of_interest(houged, np.asarray(dst_points))
    cv2.imshow("window", thresholded)
    cv2.waitKey(1000)


def main():
    # opening the video
    cap = cv2.VideoCapture(VIDEO_PATH)

    # error opening the video
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
        exit(1)
    # initialize frame size variables
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    count = 0
    # while the video is still running
    while(cap.isOpened()):
        # read each frame
        ret, frame = cap.read()
        if ret == True and count % 50 == 0:
            # detect the lanes on the frame
            detect_lanes(frame)
        count += 1
    cap.release()
    cv2.destroyAllWindows()
    # out = cv2.VideoWriter('output.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
    # for frame in frames:
    #     detect_lanes(frame, out)
    # out.release()

if __name__ == '__main__':
    argv = sys.argv
    global VIDEO_PATH
    if (len(argv) != 2):
        print("Usage: ADSDetection.py (video_path)")
        exit(1)
    else:
        VIDEO_PATH = argv[1]
    main()
