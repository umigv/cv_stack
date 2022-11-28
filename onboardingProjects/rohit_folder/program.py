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

lines = cv2.HoughLinesP(
    cropped_image,
    rho=6,
    theta=np.pi / 60,
    threshold=160,
    lines=np.array([]),
    minLineLength=40,
    maxLineGap=25
)
line_image = draw_lines(image, lines) # <---- Add this call.


cv2.imshow("image",line_image)
cv2.waitKey(4000)
