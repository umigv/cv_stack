#!/usr/bin/env python
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# from tkinter.tix import IMAGE
import cv2
import numpy as np

def main():
    # reading in an image
    image = cv2.imread('../LaneImage.jpg')
    print (type(image))
    # printing out some stats and plotting the image
    print('This image is:', type(image), 'with dimensions:', image.shape)
    cv2.imshow("Name", image)
    cv2.waitKey(8000)

    def region_of_interest(img, vertices):
    # Define a blank matrix that matches the image height/width.
    mask = np.zeros_like(img)    # Retrieve the number of color channels of the image.
    channel_count = img.shape[2]    # Create a match color with the same color channel counts.
    match_mask_color = (255,) * channel_count
      
    # Fill inside the polygon
    cv2.fillPoly(mask, vertices, match_mask_color)
    
    # Returning the image only where mask pixels match
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
    
if __name__== '__main__':
    main()