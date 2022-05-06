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
from rospy_tutorials.msg import Floats
import message_filters
import ros_numpy

# Kayla 

class kaylaLane:
    def returnLaneTest(self):
        return lanes

    def grey(self, image):
    #convert to grayscale
        image = np.asarray(image)
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    #Apply Gaussian Blur --> Reduce noise and smoothen image
    def gauss(self, image):
        return cv2.GaussianBlur(image, (5, 5), 0)

    #outline the strongest gradients in the image --> this is where lines in the image are
    def canny(self, image):
        edges = cv2.Canny(image,50,150)
        return edges

    def region(self, image):
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

    def display_lines(self, image, lines):
        lines_image = np.zeros_like(image)
        #make sure array isn't empty
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line
                #draw lines on a black image
                cv2.line(lines_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
        return lines_image

    def average(self, image, lines):
        left = []
        right = []

        if lines is not None:
            for line in lines:
                #print(line)
                x1, y1, x2, y2 = line.reshape(4)
                #fit line to points, return slope and y-int
                parameters = np.polyfit((x1, x2), (y1, y2), 1)
                #print(parameters)
                slope = parameters[0]
                y_int = parameters[1]
                #lines on the right have positive slope, and lines on the left have neg slope
                if slope < 0:
                    left.append((slope, y_int))
                else:
                    right.append((slope, y_int))
        
        #if empty, add arbitrary points
        while len(right) < 2:
            right.append([0,image.shape[0]])
        while len(left) < 2:
            left.append([0,image.shape[0]])
        #takes average among all the columns (column0: slope, column1: y_int)
        right_avg = np.average(right, axis=0)
        left_avg = np.average(left, axis=0)
        #create lines based on averages calculates
        left_line = self.make_points(image, left_avg)
        right_line = self.make_points(image, right_avg)
        return np.array([left_line, right_line])

    def make_points(self, image, average):
        #print(average)
        slope, y_int = average
        y1 = image.shape[0]
        #how long we want our lines to be --> 3/5 the size of the image
        y2 = int(y1 * (3/5))
        #determine algebraically
        if slope != 0:
            x1 = int((y1 - y_int) // slope)
            x2 = int((y2 - y_int) // slope)
            return np.array([x1, y1, x2, y2])
        else:
        #if the slope equals zero, return all zeroes
            return np.array([0, 0, 0, 0])

    #from google.colab.patches import cv2_imshow

    #Lane Detection Code:
    def __init__(self, image): 

        """
        # runtime_parameters = sl.RuntimeParameters()
        while True:
            try:
                grab = cam.grab()
                print(type(grab))
                #cv2.imshow("Image", grab)
                #cv2.waitKey(1)

                raw_image = grab
                output = raw_image.copy()

                #   Perspective Transform
                #source_points = np.float32([[580, 460], [205, 720], [1110, 720], [703, 460]]) #Old points
                #destination_points = np.float32([[320, 0], [320, 720], [960, 720], [960, 0]]) #Old points
                #source_points = np.float32([[400, 400], [100, 500], [1200, 500], [820, 400]]) #new points (1)
                #destination_points = np.float32([[0, 0], [100, 500], [1200, 500], [1200, 0]]) #new points (1)
                source_points = np.float32([[518, 485], [750, 500], [402, 555], [865, 570]]) #new points (2) works for ~44.svo
                destination_points = np.float32([[0, 0], [1200, 0], [0, 500], [1200, 500]]) #new points (2)

                shape = (raw_image.shape[1], raw_image.shape[0])
                warp_matrix = cv2.getPerspectiveTransform(source_points, destination_points)
                warp_image = cv2.warpPerspective(raw_image, warp_matrix, shape, flags = cv2.INTER_LINEAR)

                display_images = np.concatenate((warp_image, raw_image), axis = 1)

                #Display the warped image
                cv2.imshow("Warped Image", warp_image) #Display Warped Image
                cv2.waitKey(1)

                #Display the two images wide by side
                #cv2.imshow("Image", display_images)
                #cv2.waitKey(1)

                #EDGE DETECTION CODE:

                #IMAGE LOADING
                #image_path = r"/content/road_lane.jpeg"
                #image1 = cv2.imread(image_path)
                #output = image.copy()
                #plt.imshow(image1)

                '''##### DETECTING lane lines in image ######'''

                copy = np.copy(raw_image)
                edges = cv2.Canny(copy,50,150)
                isolated = region(edges)
                gray = grey(raw_image)
                cv2.imshow("edges", edges)
                cv2.imshow("isolated", isolated)
                cv2.waitKey(0)
                #plt.imshow(raw_image) #Use when finding coordinates on the raw image
                #plt.show()

                #DRAWING LINES: (order of params) --> region of interest, bin size (P, theta), min intersections needed, placeholder array, 
                lines = cv2.HoughLinesP(isolated, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
                circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100)
                averaged_lines = average(copy, lines)
                black_lines = display_lines(copy, averaged_lines)
                #taking wighted sum of original image and lane lines image
                lanes = cv2.addWeighted(copy, 0.8, black_lines, 1, 1)
                # cv2.imshow("lane window", lanes)
                # cv2.waitKey(0)
                cv2.imshow("output", lanes)
                cv2.waitKey(0)
                
                #ensure at least some circles were found
                if circles is not None:
                    # convert the (x, y) coordinates and radius of the circles to integers
                    circles = np.round(circles[0, :]).astype("int")
                    # loop over the (x, y) coordinates and radius of the circles
                    for (x, y, r) in circles:
                        # draw the circle in the output image, then draw a rectangle
                        # corresponding to the center of the circle
                        cv2.circle(output, (x, y), r, (0, 255, 0), 4)
                        #cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
                    # show the output image
                    lane_and_circle = np.concatenate((lanes, output), axis = 1)
                    cv2.imshow("output", lane_and_circle)
                    cv2.waitKey(0)
                else:
                    cv2.imshow("lane window", lanes)
                    cv2.waitKey(0)


            except ZEDRecording.GrabError:
                print("Video ended")
                break
        """

        """

        """
        
        #Process 1 image
        
        #cv2.imshow("Image", grab)
        #cv2.waitKey(1)

        raw_image = image
        output = raw_image.copy()

        #   Perspective Transform
        #source_points = np.float32([[580, 460], [205, 720], [1110, 720], [703, 460]]) #Old points
        #destination_points = np.float32([[320, 0], [320, 720], [960, 720], [960, 0]]) #Old points
        #source_points = np.float32([[400, 400], [100, 500], [1200, 500], [820, 400]]) #new points (1)
        #destination_points = np.float32([[0, 0], [100, 500], [1200, 500], [1200, 0]]) #new points (1)
        source_points = np.float32([[518, 485], [750, 500], [402, 555], [865, 570]]) #new points (2) works for ~44.svo
        destination_points = np.float32([[0, 0], [1200, 0], [0, 500], [1200, 500]]) #new points (2)
        source_points = np.float32([[518, 485], [745, 495], [400, 530], [825, 540]]) #new points (3) works for ~44.svo
        destination_points = np.float32([[0, 0], [1200, 0], [0, 500], [1200, 500]]) #new points (3)

        shape = (raw_image.shape[1], raw_image.shape[0])
        warp_matrix = cv2.getPerspectiveTransform(source_points, destination_points)
        warp_image = cv2.warpPerspective(raw_image, warp_matrix, shape, flags = cv2.INTER_LINEAR)

        display_images = np.concatenate((warp_image, raw_image), axis = 1)

        #Display the warped image
        #cv2.imshow("Warped Image", warp_image) #Display Warped Image
        #cv2.imwrite("laneTestWarped.jpg", warp_image) #Save image
        #cv2.waitKey(1)

        #Display the two images wide by side
        #cv2.imshow("Image", display_images)
        #cv2.waitKey(1)

        #EDGE DETECTION CODE:

        #IMAGE LOADING
        #image_path = r"/content/road_lane.jpeg"
        #image1 = cv2.imread(image_path)
        #output = image.copy()
        #plt.imshow(image1)

        '''##### DETECTING lane lines in image ######'''

        copy = np.copy(raw_image)
        edges = cv2.Canny(copy,50,150)
        isolated = self.region(edges)
        gray = self.grey(raw_image)
        gray_warped = self.grey(warp_image)
        #cv2.imshow("edges", edges)
        #cv2.imshow("isolated", isolated)
        #cv2.imwrite("laneTestEdges.jpg", edges)
        #cv2.imwrite("laneTestIsolated.jpg", isolated)
        #cv2.waitKey(0)
        #plt.imshow(raw_image)
        #plt.show()

        #DRAWING LINES: (order of params) --> region of interest, bin size (P, theta), min intersections needed, placeholder array, 
        lines = cv2.HoughLinesP(isolated, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
        circles = cv2.HoughCircles(gray_warped, cv2.HOUGH_GRADIENT, 1, 10)
        averaged_lines = self.average(copy, lines)
        black_lines = self.display_lines(copy, averaged_lines)
        #taking wighted sum of original image and lane lines image
        global lanes
        lanes = cv2.addWeighted(copy, 0.8, black_lines, 1, 1)
        # cv2.imshow("lane window", lanes)
        # cv2.waitKey(0)
        #cv2.imshow("output", lanes)
        #cv2.waitKey(0)

        #cv2.imwrite("laneTestOutput.jpg", lanes)
        
        #ensure at least some circles were found
        if circles is not None:
            # convert the (x, y) coordinates and radius of the circles to integers
            circles = np.round(circles[0, :]).astype("int")
            # loop over the (x, y) coordinates and radius of the circles
            for (x, y, r) in circles:
                # draw the circle in the output image, then draw a rectangle
                # corresponding to the center of the circle
                cv2.circle(output, (x, y), r, (0, 255, 0), 4)
                #cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
            # show the output image
            lane_and_circle = np.concatenate((lanes, output), axis = 1)
            #cv2.imshow("output", lane_and_circle)
            #cv2.waitKey(0)
            #print("Circle found")
            #cv2.imwrite("laneTestOtputLaneNCircle.jpg", lane_and_circle)
        #else:
            #cv2.imshow("lane window", lanes)
            #cv2.waitKey(0)
            #print("Circle Not Found")
            #cv2.imwrite("laneTestLaneWindow.jpg", lanes)

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
        params_w.minCircularity = 0.3
        
        # Set Convexity filtering parameters
        params_w.filterByConvexity = True
        params_w.minConvexity = 0.5
            
        # Set inertia filtering parameters
        params_w.filterByInertia = True
        params_w.minInertiaRatio = 0.03
        
        # Create a detector with the parameters
        detector_w = cv2.SimpleBlobDetector_create(params_w)

        #   Detect blobs for perspective transformed image
        warp_image = cv2.GaussianBlur(warp_image, (5,5), 0)
        warp_image = cv2.GaussianBlur(warp_image, (5,5), 0) #blur again
        warp_image = cv2.GaussianBlur(warp_image, (5,5), 0) #blur again
        warp_image = cv2.GaussianBlur(warp_image, (5,5), 0) #blur again
        warp_image = cv2.GaussianBlur(warp_image, (5,5), 0) #blur again
        warp_image = cv2.GaussianBlur(warp_image, (5,5), 0) #blur again
        warp_image = cv2.addWeighted(warp_image, 1.3, warp_image, 0, 3) #Increase contrast
        #cv2.imwrite("laneTestWarpBlurred.jpg", warp_image)

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
        #cv2.imwrite("laneTestBlob.jpg", blobs)
        #cv2.imwrite("laneTestLaneNBlob.jpg", lanes)
        #cv2.imwrite("laneTestWarpBlob.jpg", blobsWarp)

        print(number_of_blobs, " blobs found")









# Andrew and David (and Sid)

class ADSDetection:

    def returnEDImage(self):
        return edgeDetectedImage

    def returnHougedImage(self):
        return houged
    
    def returnCroppedImage(self):
        return newImg

    def __init__(self, image, dst_points):
        # mask non-white to black
        white_mask = self.white_mask(image)

        # grayscale the image
        grayscaled = self.grayscale(image)

        # apply gaussian blur
        kernelSize = 5
        gaussianBlur = self.gaussian_blur(grayscaled, kernelSize)

        # canny
        minThreshold = 100
        maxThreshold = 200
        global edgeDetectedImage
        edgeDetectedImage = cv2.Canny(gaussianBlur, minThreshold, maxThreshold)
        #apply mask
        # lowerLeftPoint = [130, 540]
        # upperLeftPoint = [410, 350]
        # upperRightPoint = [570, 350]
        # lowerRightPoint = [915, 540]

        # pts = np.array([[lowerLeftPoint, upperLeftPoint, upperRightPoint,
        # lowerRightPoint]], dtype=np.int32)
        # masked_image = region_of_interest(edgeDetectedImage, pts)


        #hough lines
        rho = 1
        theta = np.pi/180
        threshold = 30
        min_line_len = 20
        max_line_gap = 20
        global houged
        houged = self.hough_lines(edgeDetectedImage, rho, theta,
                        threshold, min_line_len, max_line_gap)
        # bottomLeft, bottomRight, topLeft, topRight = dst_quad
        bottomLeft, bottomRight, topLeft, topRight = dst_points
        bottomLeft = [bottomLeft[0]+5, bottomLeft[1]]
        bottomRight = [bottomRight[0]-5, bottomRight[1]]
        topLeft = [topLeft[0]+5, topLeft[1]]
        topRight = [topRight[0]-5, topRight[1]]
        # needed to flip bottomRight and bottomLeft for fillPoly function to work correctly
        dst_points = [bottomRight, bottomLeft, topLeft, topRight]
        rospy.loginfo(np.asarray(dst_points))
        global newImg
        newImg = self.region_of_interest(edgeDetectedImage, np.asarray(dst_points))
        

    def grayscale(self,img):
        """Applies the Grayscale transform
        This will return an image with only one color channel
        but NOTE: to see the returned image as grayscale
        you should call plt.imshow(gray, cmap='gray')"""
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def white_mask(self, img):
        # https://stackoverflow.com/questions/22588146/tracking-white-color-using-python-opencv
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # define range of white color in HSV
        # change it according to your need !
        lower_white = np.array([0,0,0], dtype=np.uint8)
        upper_white = np.array([0,0,255], dtype=np.uint8)
        # Threshold the HSV image to get only white colors
        mask = cv2.inRange(hsv, lower_white, upper_white)
        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(img,img, mask= mask)
        return res

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
        rospy.loginfo(type(lines))
        rospy.loginfo(lines)
        if isinstance(lines, np.ndarray):
            rospy.loginfo(lines)
            self.draw_lines(line_img, lines)
        return line_img


    def draw_lines(self,img, lines, color=[255, 0, 0], thickness=2):
        """
        This function draws `lines` with `color` and `thickness`.
        """
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)



def callback(data, dst):
    rospy.loginfo("Converting perspective transformed img to edge detection image")
    bridge = CvBridge()
    timestamp = data.header.stamp
    cv_image = bridge.imgmsg_to_cv2(data, desired_encoding='bgra8')
    # import pdb; pdb.set_trace()
    dst = ros_numpy.numpify(dst)
    ads = ADSDetection(cv_image, dst)
    kayla = kaylaLane(cv_image)
    # publish_transform(bridge, "/cv/laneMapping/left", ads.returnCroppedImage(), timestamp)
    publish_transform(bridge, "/cv/laneMapping/left", kayla.returnLaneTest(), timestamp)

def publish_transform(bridge, topic, cv2_img, timestamp):
    transformed_ros = bridge.cv2_to_imgmsg(cv2_img)
    transformed_ros.header.stamp = timestamp
    pub = rospy.Publisher(topic, Image, queue_size=10)
    pub.publish(transformed_ros)
    rospy.loginfo("Published")

def listener():
    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('ads_detection', anonymous=True)
    rospy.loginfo("Initialized ADS Detection")

    image_sub = message_filters.Subscriber("/cv/perspective/left", Image)
    dst_sub = message_filters.Subscriber("/cv/perspective/dst_quad", Image)
    
    ts = message_filters.TimeSynchronizer([image_sub, dst_sub], 10)
    ts.registerCallback(callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


if __name__ == '__main__':
    try:
        listener()
    except rospy.ROSInterruptException:
        pass
