# Add all the necessary libraries
import sys
import pyzed.sl as sl
import cv2
import time
import numpy as np
import matplotlib as plt
from zedRecording import ZEDRecording
from ADSDetection import ADSDetection

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
            #cv2.imshow("RawImage", grab)
            #cv2.waitKey(0)

            edges = ADSDetection(grab)
            print(type(edges))
            print(type(edges.returnHougedImage()))
            cv2.imshow("EdgesImage", edges.returnEDImage())
            cv2.waitKey(0)
            cv2.imshow("EdgesImage", edges.returnHougedImage())
            cv2.waitKey(0)

            # Perspective Transform
            #source_points = np.float32([[580, 460], [205, 720], [1110, 720], [703, 460]]) #Old points
            #destination_points = np.float32([[320, 0], [320, 720], [960, 720], [960, 0]]) #Old points
            source_points = np.float32([[400, 400], [100, 500], [1200, 500], [820, 400]]) #new points
            destination_points = np.float32([[0, 0], [100, 500], [1200, 500], [1200, 0]]) #new points

            shape = (grab.shape[1], grab.shape[0])
            warp_matrix = cv2.getPerspectiveTransform(source_points, destination_points)
            warp_image = cv2.warpPerspective(edges, warp_matrix, shape, flags = cv2.INTER_LINEAR)

            display_images = np.concatenate((warp_image, edges), axis = 1)

            #Display the warped image
            cv2.imshow("Warped Image", warp_image)
            cv2.waitKey(1)

            #Display the two images wide by side
            cv2.imshow("Image", display_images)
            cv2.waitKey(1)

        except ZEDRecording.GrabError:
            print("Video ended")
            break

    cam.close()

if __name__ == "__main__":
    main()