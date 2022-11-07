import cv2
import numpy as np


# reading in an image
image = cv2.imread('../LaneImage.jpg')
# printing out some stats and plotting the image
height, width, random = image.shape
print('This image is:', type(image), 'with dimensions:', image.shape)

region_of_interest_vertices = [
    (0, height),
    (width / 2, height / 2),
    (width, height),
]

def region_of_interest(img, vertices):
    # Define a blank matrix that matches the image height/width.
    mask = np.zeros_like(img)
    # Retrieve the number of color channels of the image.
    # Create a match color with the same color channel counts.
    match_mask_color = 255    

    # Fill inside the polygon
    cv2.fillPoly(mask, vertices, match_mask_color)
    
    # Returning the image only where mask pixels match
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image



gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# Call Canny Edge Detection here.

cannyed_image = cv2.Canny(gray_image, 100, 200)


cropped_image = region_of_interest(
    cannyed_image,
    np.array([region_of_interest_vertices], np.int32),
)


cv2.imshow("image",cropped_image)
cv2.waitKey(4000)
