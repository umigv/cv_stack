"""
Resources used: 
    - https://medium.com/@heeduugar/canny-edge-detector-in-python-and-opencv-cpu-vs-gpu-with-cuda-229e7bafc5e9
    - https://learnopencv.com/getting-started-opencv-cuda-module/
    
"""
import numpy as np
import cv2 as cv
from cv2 import cuda
from matplotlib import pyplot as plt

img = cv.imread('LaneImage.jpg',0)
imgMat = cv.cuda_GpuMat(img)
detector = cv.cuda.createCannyEdgeDetector(low_thresh=100, high_thresh=110)
dstImg = detector.detect(imgMat)
canny = dstImg.download()
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(canny,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()


"""
import cv2
img = cv2.imread("LaneImage.jpg", cv2.IMREAD_GRAYSCALE)
src = cv2.cuda_GpuMat()
src.upload(img)
clahe = cv2.cuda.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
dst = clahe.apply(src, cv2.cuda_Stream.Null())
result = dst.download()
cv2.imshow("result", result)
cv2.waitKey(0)
"""