#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import sys

class ADSDetection:

    def returnEDImage(self):
        return edgeDetectedImage

    def returnHougedImage(self):
        return houged

    def __init__(self, image):

        # grayscale the image
        grayscaled = self.grayscale(image)
        # cv2.imshow("grayscale", grayscaled)
        # cv2.waitKey(0)
        # apply gaussian blur
        kernelSize = 5
        gaussianBlur = self.gaussian_blur(grayscaled, kernelSize)

        # canny
        minThreshold = 100
        maxThreshold = 200
        global edgeDetectedImage
        edgeDetectedImage = cv2.Canny(gaussianBlur, minThreshold, maxThreshold)
        # cv2.imshow("edges", edgeDetectedImage)
        # cv2.waitKey(0)
        #apply mask
        # lowerLeftPoint = [130, 540]
        # upperLeftPoint = [410, 350]
        # upperRightPoint = [570, 350]
        # lowerRightPoint = [915, 540]

        # pts = np.array([[lowerLeftPoint, upperLeftPoint, upperRightPoint, 
        # lowerRightPoint]], dtype=np.int32)
        # masked_image = region_of_interest(edgeDetectedImage, pts)

        # cv2.imshow("masked", masked_image)
        # cv2.waitKey(0)
        #hough lines
        rho = 1
        theta = np.pi/180
        threshold = 30
        min_line_len = 20 
        max_line_gap = 20
        global houged
        houged = self.hough_lines(edgeDetectedImage, rho, theta, 
                        threshold, min_line_len, max_line_gap)

        # cv2.imshow("houged", houged)
        # cv2.waitKey(0)

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
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        
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

        line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
        
        self.draw_lines(line_img, lines)
        return line_img

    def draw_lines(self,img, lines, color=[255, 0, 0], thickness=2):
        """
        This function draws `lines` with `color` and `thickness`.    
        """
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    

    




