import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

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


image = mpimg.imread('lanes.jpg')

h, w, c = image.shape
print('width:  ', w)
print('height: ', h)
print('channel:', c)

region_of_interest_vertices = [
    (0, h),
    (w / 2, h / 2),
    (w, h),
]

cropped_image = region_of_interest(
    image, np.array([region_of_interest_vertices], np.int32),)

plt.figure()
plt.imshow(cropped_image)

# reading in an image
# printing out some stats and plotting the image
plt.imshow(image)
plt.imshow(cropped_image)
plt.show()
print('This image is:', type(image), 'with dimensions:', image.shape)