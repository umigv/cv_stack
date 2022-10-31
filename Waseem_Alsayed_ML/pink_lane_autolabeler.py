import cv2
# from google.colab.patches import cv2_imshow
# from google.colab import drive
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

# https://stackoverflow.com/questions/22704936/reading-every-nth-frame-from-videocapture-in-opencv

# os.mount('/catkin_ws/src/cv_stack/') #('/content/drive')
# Find video file
#!ls "/content/drive/Shareddrives/Robotics | UMARV/2022 - 2023/Computer Vision/Machine Learning/Raw Video/"

#Get frames from mp4
cap = cv2.VideoCapture("video3.MOV")
fps = cap.get(cv2.CAP_PROP_FPS)

print(fps)

frames = []

n = 1
while(cap.isOpened()):
  is_read, frame = cap.read()
  if not is_read:
    break

  # if n%5==0:
  #   frames.append(frame)
  # if n > 101:
  #     break
  n = n + 1

  image = frame
 
  # Converting the image to hsv
  hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
  
  # define range of red color in HSV
  lower_pink = np.array([140,25,120])
  upper_pink = np.array([170,255,255])
      
  # Threshold the HSV image using inRange function to get only pink colors
  mask = cv2.inRange(hsv, lower_pink, upper_pink)
  
  if n % 10 == 0:
    # plt.figure(figsize=[30,30])
    cv2.imshow("original", image)
    cv2.imshow("lanes", mask)
    cv2.waitKey(1)
    # plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Original Image",fontdict={'fontsize': 25});plt.axis('off');
    # plt.subplot(122);plt.imshow(mask, cmap='gray');plt.title("Mask of pink Color",fontdict={'fontsize': 25});plt.axis('off');


# for i in range(10):
  # cv2_imshow(frames[i*10])