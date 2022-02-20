import sys
import pyzed.sl as sl
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from zedRecording import ZEDRecording

def grey(image):
  #convert to grayscale
    image = np.asarray(image)
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

  #Apply Gaussian Blur --> Reduce noise and smoothen image
def gauss(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

  #outline the strongest gradients in the image --> this is where lines in the image are
def canny(image):
    edges = cv2.Canny(image,50,150)
    return edges

def region(image):
    height, width = image.shape
    #isolate the gradients that correspond to the lane lines
    # triangle = np.array([
    #                    [(100, height), (475, 325), (width, height)]
    #                    ])

    #THIS WORKS
    triangle = np.array([
                       [(250, 550), (663, 463), (1055, 576)]
                       ])
    
    # triangle = np.array([
    #                    [(652, 570), (740, 482), (936, 576)]
    #                    ])
    #create a black image with the same dimensions as original image
    mask = np.zeros_like(image)
    #create a mask (triangle that isolates the region of interest in our image)
    mask = cv2.fillPoly(mask, triangle, 255)
    mask = cv2.bitwise_and(image, mask)
    return mask

def display_lines(image, lines):
    lines_image = np.zeros_like(image)
    #make sure array isn't empty
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line
            #draw lines on a black image
            cv2.line(lines_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return lines_image

def average(image, lines):
    left = []
    right = []

    if lines is not None:
      for line in lines:
        print(line)
        x1, y1, x2, y2 = line.reshape(4)
        #fit line to points, return slope and y-int
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        print(parameters)
        slope = parameters[0]
        y_int = parameters[1]
        #lines on the right have positive slope, and lines on the left have neg slope
        if slope < 0:
            left.append((slope, y_int))
        else:
            right.append((slope, y_int))
            
    #takes average among all the columns (column0: slope, column1: y_int)
    right_avg = np.average(right, axis=0)
    left_avg = np.average(left, axis=0)
    #create lines based on averages calculates
    left_line = make_points(image, left_avg)
    right_line = make_points(image, right_avg)
    return np.array([left_line, right_line])

def make_points(image, average):
    print(average)
    slope, y_int = average
    y1 = image.shape[0]
    #how long we want our lines to be --> 3/5 the size of the image
    y2 = int(y1 * (3/5))
    #determine algebraically
    x1 = int((y1 - y_int) // slope)
    x2 = int((y2 - y_int) // slope)
    return np.array([x1, y1, x2, y2])

#from google.colab.patches import cv2_imshow

#Lane Detection Code:
def main(): 
    if len(sys.argv) != 2:
        print("Please specify path to .svo file.")
        exit()
        
    filepath = sys.argv[1]
    print("Reading SVO file: {0}".format(filepath))

    cam = ZEDRecording(filepath)
    
    #Process 1 image
    
    grab = cam.grab()
    print(type(grab))
    raw_image = grab
    output = raw_image.copy()

    '''##### DETECTING lane lines in image ######'''
    copy = np.copy(raw_image)
    edges = cv2.Canny(copy,50,150)
    isolated = region(edges)
    gray = grey(raw_image)
    gray_warped = grey(warp_image)
    cv2.imwrite("laneTestEdges.jpg", edges)
    cv2.imwrite("laneTestIsolated.jpg", isolated)

    #DRAWING LINES: (order of params) --> region of interest, bin size (P, theta), min intersections needed, placeholder array, 
    lines = cv2.HoughLinesP(isolated, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    # circles = cv2.HoughCircles(gray_warped, cv2.HOUGH_GRADIENT, 1, 10)
    averaged_lines = average(copy, lines)
    black_lines = display_lines(copy, averaged_lines)
    #taking wighted sum of original image and lane lines image
    lanes = cv2.addWeighted(copy, 0.8, black_lines, 1, 1)
    # cv2.imshow("lane window", lanes)
    # cv2.waitKey(0)
    #cv2.imshow("output", lanes)
    #cv2.waitKey(0)

    cv2.imwrite("laneTestOutput.jpg", lanes)
    #END

    #NEW: Blob Detection

    params = cv2.SimpleBlobDetector_Params()
    
    # Set Area filtering parameters
    params.filterByArea = True
    params.minArea = 40
    
    # Set Circularity filtering parameters
    params.filterByCircularity = True
    params.minCircularity = 0.4
    
    # Set Convexity filtering parameters
    params.filterByConvexity = True
    params.minConvexity = 0.8
        
    # Set inertia filtering parameters
    params.filterByInertia = True
    params.minInertiaRatio = 0.04
    
    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)
        
    # Detect blobs
    keypoints = detector.detect(isolated)


    #       WARPED PARAMS
    params_w = cv2.SimpleBlobDetector_Params()
    
    # Set Area filtering parameters
    params_w.filterByArea = False
    params_w.minArea = 2
    
    # Set Circularity filtering parameters
    params_w.filterByCircularity = True
    params_w.minCircularity = 0.4
    
    # Set Convexity filtering parameters
    params_w.filterByConvexity = True
    params_w.minConvexity = 0.5
        
    # Set inertia filtering parameters
    params_w.filterByInertia = True
    params_w.minInertiaRatio = 0.03
    
    # Create a detector with the parameters
    detector_w = cv2.SimpleBlobDetector_create(params_w)

    #   Detect blobs for perspective transformed image
    warp_image = cv2.GaussianBlur(gray_warped, (5,5), 0)
    # warp_image = cv2.GaussianBlur(warp_image, (5,5), 0) #blur again
    # warp_image = cv2.GaussianBlur(warp_image, (5,5), 0) #blur again
    # warp_image = cv2.GaussianBlur(warp_image, (5,5), 0) #blur again
    # warp_image = cv2.GaussianBlur(warp_image, (5,5), 0) #blur again
    # warp_image = cv2.GaussianBlur(warp_image, (5,5), 0) #blur again
    warp_image = cv2.addWeighted(warp_image, 1.3, warp_image, 0, 3) #Increase contrast
    cv2.imwrite("laneTestWarpBlurred.jpg", warp_image)

    warp_edges = cv2.Canny(warp_image, 20, 50)
    warp_edges = cv2.GaussianBlur(warp_edges, (5,5), 0) #Blur edges
    warp_edges = cv2.GaussianBlur(warp_edges, (5,5), 0) #Blur edges
    warp_edges = cv2.GaussianBlur(warp_edges, (5,5), 0) #Blur edges

    warp_edges = cv2.inRange(warp_edges, 1, 255)

    #warp_edges = cv2.bitwise_and(warp_edges, warp_edges, mask=mask)

    keypointsWarp = detector_w.detect(warp_edges)
    
    # Draw blobs on our image as red circles
    blank = np.zeros((1, 1))
    blobs = cv2.drawKeypoints(isolated, keypoints, blank, (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    blobsWarp = cv2.drawKeypoints(warp_edges, keypointsWarp, blank, (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    for point in keypoints:
        (x, y) = point.pt
        cv2.circle(lanes,(round(x), round(y)), 20, (0,255,0), 3)
    
    for point in keypointsWarp:
        (x, y) = point.pt
        cv2.circle(blobsWarp,(round(x), round(y)), 20, (0,255,0), 3)
    
    number_of_blobs = len(keypoints)
    text = "Number of Circular Blobs: " + str(number_of_blobs)
    text_warp = "Number of Circular Blobs (Warped): " + str(len(keypointsWarp))
    cv2.putText(blobs, text, (20, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)
    cv2.putText(lanes, text, (20, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)
    cv2.putText(blobsWarp, text_warp, (20, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)
    
    # Show blobs
    cv2.imwrite("laneTestBlob.jpg", blobs)
    cv2.imwrite("laneTestLaneNBlob.jpg", lanes)
    cv2.imwrite("laneTestWarpBlob.jpg", blobsWarp)

    print(number_of_blobs, " blobs found")

    cam.close()

if __name__ == "__main__":
    main()