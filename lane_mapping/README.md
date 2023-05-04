# lane_mapping

## Synopsis

`lane_mapping` is the main ROS package that we will be using in the 2023 competition. It contains three source files:

## LaneDetection.py
LaneDetection is our main lane mapping node currently. It listens to the grayscale left camera image (`/zed/zed_node/left/image_rect_gray`) from the Zed camera, the depth map (`/zed/zed_node/depth/depth_registered`), and the depth camera info topic (`/zed/zed_node/depth/camera_info`). It outputs the depth image after running our lane detection to `/cv/laneMapping/left` and `/cv/laneMapping/camera_info`. It does not make any changes to the camera info topic, other than changing the name so that the launch file will work correctly. The lane detection output is the result of running the lane detection, then using that as a mask for the depth map, and should leave only depth values where lanes are detected.

## CudaDetection.py

CudaDetection is an attempt at converting our current LaneDetection code into code that utilizes the Cuda library. It contains the same functions as LaneDetection.py, but those functions use the Cuda library. This has not yet been updated to run correctly with our full pipeline, but does have functions for lane detection similar to our LaneDetection.py file

## OldLaneDetection.py

This is the file that contains certain functions for lane detection that are not kept anywhere else. Since we wanted to clean up the LaneDetection.py file, we removed some functions that were not currently being used but might have been useful in the future. These functions should be contained within this file, but otherwise LaneDetection.py should be used for our actual pipeline.

## ADSDetection.py

ADSDetection is our legacy file. It was written for the 2022 competition and has not been edited much since then. It most likely will not work with the new Jetson Orin, since the Orin can run newer versions of ROS and OpenCV. This file is useful to see syntax for listening to multiple topics in ROS, and it is useful to see what we ran in 2022.