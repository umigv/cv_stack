# Add all the necessary libraries
import sys
import pyzed.sl as sl
import cv2
import time
import numpy as np
import matplotlib as plt
from zedRecording import ZEDRecording

def main(): 
    if len(sys.argv) != 2:
        print("Please specify path to .svo file.")
        exit()
        
    filepath = sys.argv[1]
    print("Reading SVO file: {0}".format(filepath))

    cam = ZEDRecording(filepath)

    # runtime_parameters = sl.RuntimeParameters()
    while True:
        try:
            grab = cam.grab()
            print(type(grab))
            #cv2.imshow("Image", grab)
            #cv2.waitKey(1)

            raw_image = grab

            #   Perspective Transform
            #source_points = np.float32([[580, 460], [205, 720], [1110, 720], [703, 460]]) #Old points
            #destination_points = np.float32([[320, 0], [320, 720], [960, 720], [960, 0]]) #Old points
            source_points = np.float32([[400, 400], [100, 500], [820, 400], [1200, 500]]) #new points
            destination_points = np.float32([[0, 0], [100, 500], [1200, 0], [1200, 500]]) #new points

            shape = (raw_image.shape[1], raw_image.shape[0])
            warp_matrix = cv2.getPerspectiveTransform(source_points, destination_points)
            warp_image = cv2.warpPerspective(raw_image, warp_matrix, shape, flags = cv2.INTER_LINEAR)

            display_images = np.concatenate((warp_image, raw_image), axis = 1)

            #Display the warped image
            #plt.imshow(warp_image)
            #plt.show()
            cv2.imshow("Warped Image", warp_image)
            cv2.waitKey(1)

            #Display the two images wide by side
            #cv2.imshow("Image", display_images)
            #cv2.waitKey(1)

        except ZEDRecording.GrabError:
            print("Video ended")
            break

    cam.close()

if __name__ == "__main__":
    main()

        
