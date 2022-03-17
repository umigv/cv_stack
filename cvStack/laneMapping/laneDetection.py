# Add all the necessary libraries
import sys
import pyzed.sl as sl
import cv2
import time
import numpy as np
import matplotlib as plt
from zedRecording import ZEDRecording
from ADSDetection import ADSDetection
from kaylaLaneDetectionCopy import kaylaLane

def main():
    if len(sys.argv) != 2:
        print("Please specify path to .svo file.")
        exit()
        
    filepath = sys.argv[1]
    print("Reading SVO file: {0}".format(filepath))

    cam = ZEDRecording(filepath)

    # runtime_parameters = sl.RuntimeParameters()
    frames = []
    run = True
    while True:
        try:
            if run :
                grab = cam.grab()
                print(type(grab))
                #cv2.imshow("RawImage", grab)
                #cv2.waitKey(0)

                #ADS Detection
                
                edges = ADSDetection(grab)
                EdImage = edges.returnEDImage()
                HoughImage = edges.returnHougedImage()
                laneImage = HoughImage

                #kaylaLaneTest (not working)
                '''
                edges = kaylaLane(grab)
                laneImage = edges.returnLaneTest()
                '''
                # cv2.imshow("EdgesImage", EdImage)
                # cv2.waitKey(1)
                # cv2.imshow("EdgesImage", HoughImage)
                # cv2.waitKey(0)
                frames.append(laneImage)

                #   Perspective Transform
                #source_points = np.float32([[580, 460], [205, 720], [1110, 720], [703, 460]]) #Old points
                #destination_points = np.float32([[320, 0], [320, 720], [960, 720], [960, 0]]) #Old points
                source_points = np.float32([[400, 400], [100, 500], [1200, 500], [820, 400]]) #new points
                destination_points = np.float32([[0, 0], [100, 500], [1200, 500], [1200, 0]]) #new points
                """
                #New Points
                source_points = np.float32([[518, 485], [750, 500], [402, 555], [865, 570]]) #new points (2) works for ~44.svo
                destination_points = np.float32([[0, 0], [1200, 0], [0, 500], [1200, 500]]) #new points (2)
                source_points = np.float32([[518, 485], [745, 495], [410, 530], [825, 540]]) #new points (3) works for ~44.svo
                destination_points = np.float32([[0, 0], [1200, 0], [0, 500], [1200, 500]]) #new points (3)
                """
                run = False
            else:
                run = True
            

            # shape = (grab.shape[1], grab.shape[0])
            # warp_matrix = cv2.getPerspectiveTransform(source_points, destination_points)
            # warp_image = cv2.warpPerspective(EdImage, warp_matrix, shape, flags = cv2.INTER_LINEAR)

            # display_images = np.concatenate((warp_image, EdImage), axis = 1)

            #Display the warped image
            # cv2.imshow("Warped Image", warp_image)
            # cv2.waitKey(1)

            #Display the two images side by side
            #cv2.imshow("Image", display_images)
            #cv2.waitKey(1)

        except ZEDRecording.GrabError:
            print("Video ended")
            break

    cam.close()
    size = frames[0].shape[:2]
    #height, width, dummy = frames[0].shape
    #size = (width,height)
    out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 30, (size[1], size[0]))
    
    
    for i in range(len(frames)):
        out.write(frames[i])
    out.release()

if __name__ == "__main__":
    main()

