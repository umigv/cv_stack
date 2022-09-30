import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
from cv2 import cuda

# reading in an image

# cuda.printCudaDeviceInfo(0)

img = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE)
src = cv2.cuda_GpuMat()
src.upload(img)

clahe = cv2.cuda.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
dst = clahe.apply(src, cv2.cuda_Stream.Null())

result = dst.download()

cv2.imshow("result", result)
cv2.waitKey(0)

# gpu_frame = cv.cuda_GpuMat()
# screenshot = cv.imread('lanes.jpg')
# gpu_frame.upload(screenshot)
# screenshot = cv.cuda.cvtColor(gpu_frame, cv.COLOR_RGB2BGR)
# screenshot = cv.cuda.resize(screenshot, (400, 400))
# screenshot.download()

# from PIL import Image
# Image.fromarray(screenshot)

'''
def region_of_interest(img, vertices):
    # Define a blank matrix that matches the image height/width.
    mask = np.zeros_like(img)
    # Retrieve the number of color channels of the image.
    #channel_count = img.shape[2]
    # Create a match color with the same color channel counts.
    #match_mask_color = (255,) * channel_count
    match_mask_color = 255 # <-- This line altered for grayscale.
     
    # Fill inside the polygon
    cv2.fillPoly(mask, vertices, match_mask_color)
   
    # Returning the image only where mask pixels match
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

image = mpimg.imread('solidWhiteCurve.jpg')

region_of_interest_vertices = [
    (0, image.shape[0]),
    (image.shape[1] / 2, image.shape[0] / 2),
    (image.shape[1], image.shape[0]),
]

# printing out some stats and plotting the image
#print('This image is:', type(image), 'with dimensions:', image.shape)
#plt.figure()
#plt.imshow(image)
#plt.show()


def draw_lines(img, lines, color=[255, 0, 0], thickness=3):
    # If there are no lines to draw, exit.
    if lines is None:
        return
    # Make a copy of the original image.
    img = np.copy(img)
    # Create a blank image that matches the original in size.
    line_img = np.zeros(
        (
            img.shape[0],
            img.shape[1],
            3
        ),
        dtype=np.uint8,
    )
    # Loop over all lines and draw them on the blank image.
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
    # Merge the image with the lines onto the original.
    img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
    # Return the modified image.
    return img


# Convert to grayscale here.
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# Call Canny Edge Detection here.
cannyed_image = cv2.Canny(gray_image, 100, 200)

cropped_image = region_of_interest(
    cannyed_image,
    np.array([region_of_interest_vertices], np.int32),
)

lines = cv2.HoughLinesP(
    cropped_image,
    rho=6,
    theta=np.pi / 60,
    threshold=160,
    lines=np.array([]),
    minLineLength=40,
    maxLineGap=25
)

left_line_x = []
left_line_y = []
right_line_x = []
right_line_y = []
for line in lines:
    for x1, y1, x2, y2 in line:
        slope = (y2 - y1) / (x2 - x1) # <-- Calculating the slope.
        if math.fabs(slope) < 0.5: # <-- Only consider extreme slope
            continue
        if slope <= 0: # <-- If the slope is negative, left group.
            left_line_x.extend([x1, x2])
            left_line_y.extend([y1, y2])
        else: # <-- Otherwise, right group.
            right_line_x.extend([x1, x2])
            right_line_y.extend([y1, y2])
           
min_y = image.shape[0] * (3 / 5) # <-- Just below the horizon
max_y = image.shape[0] # <-- The bottom of the image

poly_left = np.poly1d(np.polyfit(
    left_line_y,
    left_line_x,
    deg=1
))
left_x_start = int(poly_left(max_y))
left_x_end = int(poly_left(min_y))
poly_right = np.poly1d(np.polyfit(
    right_line_y,
    right_line_x,
    deg=1
))
right_x_start = int(poly_right(max_y))
right_x_end = int(poly_right(min_y))

line_image = draw_lines(image, lines) # <---- Add this call.
plt.figure()
plt.imshow(line_image)

plt.figure()
plt.imshow(cropped_image)

plt.show()
'''