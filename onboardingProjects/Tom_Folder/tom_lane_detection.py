import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math

# Function to show our region of interest in the image
def region_of_interest(img, vertices):
    # Defines a blank matrix that matches the image height and width
    mask = np.zeros_like(img)

    match_mask_color = 255

    # Fills inside the polygon
    cv2.fillPoly(mask, vertices, match_mask_color)

    # Returns the image only where mask pixels match
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

# Function to draw lines that make up the lanes onto the image
def draw_lines(img, lines, color = [255, 0, 0], thickness = 3):
    # If there are no lines to draw, exit
    if lines is None:
        return

    # Makes a copy of the original image
    img = np.copy(img)

    # Creates a blank image that matches the original in size
    line_img = np.zeros(
        (
            img.shape[0],
            img.shape[1],
            3
        ),
        dtype = np.uint8
    )

    # Loops over all lines and draws them on the blank image
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)

    # Merges the image with the lines ontp the original image
    img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)

    # Returns modified image
    return img

# Main function
def main():
    # Reads the unedited image
    image = cv2.imread('LaneImage.jpg')
     

    # Defines height and width
    height = image.shape[0]
    width = image.shape[1]

    # Defines region of interest
    region_of_interest_vertices = [
        (0, height),
        (width/2, height/2),
        (width, height),
    ]

    # Converts to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Calls Canny Edge Detection
    cannyed_image = cv2.Canny(gray_image, 100, 200)

    # Cropped image showing region of interest
    cropped_image = region_of_interest(cannyed_image, np.array([region_of_interest_vertices], np.int32))

    # Draws lines that make up the lanes
    lines = cv2.HoughLinesP(
        cropped_image,
        rho = 6,
        theta = np.pi / 60,
        threshold = 160,
        lines = np.array([]),
        minLineLength = 40,
        maxLineGap = 25
    )

    # List of left lines
    left_line_x = []
    left_line_y = []

    # List of right lines
    right_line_x = []
    right_line_y = []

    # Sorts the line into either left lines or right lines
    for line in lines:
        line = line[0]

        x1 = line[0]
        x2 = line[1]
        y1 = line[2]
        y2 = line[3]
        # Calculates slope
        slope = (y2 - y1) / (x2 - x1)
        # Only considers extreme slopes
        if math.fabs(slope) < 0.5:
            continue
        # If slope is negative, it is in the left group
        if slope <= 0:
            left_line_x.extend([x1, x2])
            left_line_y.extend([y1, y2])
        # Otherwise it is in the right group
        else: 
            right_line_x.extend([x1, x2])
            right_line_y.extend([y1, y2])

    # Just below the horizon
    min_y = int(image.shape[0] * (3 / 5))
    # Bottom of image
    max_y = int(image.shape[0])

    # Generates a linear function for the left lanes
    poly_left = np.poly1d(np.polyfit(
            left_line_y, 
            left_line_x, 
            deg = 1
    ))
    left_x_start = int(poly_left(max_y))
    left_x_end = int(poly_left(min_y))

    # Generates a linear function for the right lanes
    poly_right = np.poly1d(np.polyfit(
            right_line_y, 
            right_line_x, 
            deg = 1
    ))
    right_x_start = int(poly_right(max_y))
    right_x_end = int(poly_right(min_y))

    # Calls draw_lines function
    line_image = draw_lines(
            image,
            [[
                [left_x_start, max_y, left_x_end, min_y],
                [right_x_start, max_y, right_x_end, min_y]
            ]],
            thickness = 5
    )

    plt.figure()
    cv2.imshow('lane_image',line_image)
    cv2.waitKey(5000)

# Main function will run if someone calls main
if __name__ == '__main__':
    main()