import cv2
import numpy as np
# reading in an image
image = cv2.imread('LaneImage.jpg')
# printing out some stats and plotting the image
print('This image is:', type(image), 'with dimensions:', image.shape)
cv2.imshow("Window",image)
cv2.waitKey(2000)

def region_of_interest(image, vertices):
    mask = np.zeros_like(image)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=3):
    line_img = np.zeros(
        (
            img.shape[0],
            img.shape[1],
            3
        ),
        dtype=np.uint8
    )
    img = np.copy(img)
    if lines is None:
        return
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
    img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
    return img

height = image.shape[0]
width = image.shape[1]
region_of_interest_vertices = [
    (0, height),
    (width / 2, height / 2),
    (width, height),
]

gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
cannyed_image = cv2.Canny(gray_image, 100, 200)
cv2.imshow("Window", cannyed_image)
cv2.waitKey(2000)
print (cannyed_image.shape)

cropped_image = region_of_interest(
    cannyed_image,
    np.array(
     [region_of_interest_vertices],
     np.int32
    ),
)
cv2.imshow("Window", cropped_image)
cv2.waitKey(2000)

lines = cv2.HoughLinesP(
    cropped_image,
    rho=6,
    theta=np.pi / 60,
    threshold=160,
    lines=np.array([]),
    minLineLength=40,
    maxLineGap=25
)
print(lines)

line_image = draw_lines(image, lines)
cv2.imshow("Window", line_image)
cv2.waitKey(2000)
