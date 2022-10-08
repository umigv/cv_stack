import cv2
# reading in an image
image = cv2.imread('LaneImage.jpg')
# printing out some stats and plotting the image
print('This image is:', type(image), 'with dimensions:', image.shape)
cv2.imshow("Window",image)
cv2.waitKey(2000)