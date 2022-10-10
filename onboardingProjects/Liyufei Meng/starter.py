import cv2
import numpy as np
# reading in an image
image = cv2.imread('LaneImage.jpg')
# printing out some stats and plotting the image
print('This image is:', type(image), 'with dimensions:', image.shape)
cv2.imshow("Window",image)
cv2.waitKey(2000)

def region_of_interest(image, vertices):
    # Define a blank matrix that matches the image height/width.
    mask = np.zeros_like(image)
    # Retrieve the number of color channels of the image.
    channel_count = image.shape[2]
    # Create a match color with the same color channel counts.
    match_mask_color = (255,) * channel_count
      
    # Fill inside the polygon
    cv2.fillPoly(mask, vertices, match_mask_color)
    
    # Returning the image only where mask pixels match
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

height = image.shape[0]
width = image.shape[1]
region_of_interest_vertices = [
    (0, height),
    (width / 2, height / 2),
    (width, height),
]

cropped_image = region_of_interest(
    image,
    np.array([region_of_interest_vertices], np.int32),
)
cv2.imshow("Window",cropped_image)
cv2.waitKey(2000)