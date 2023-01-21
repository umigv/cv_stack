# lane_mapping

## Synopsis

`lane_mapping` is the main ROS package that we will be using in the 2023 competition. It contains three source files:

## LaneDetection.py
LaneDetection is our main lane mapping node currently. It listens to the left camera image from the Zed camera, and outputs the image after running our lane detection to `/cv/laneMapping/left`, and the occupancy grid to `/cv/laneMapping/ogrid`. Both of these should only have values where it detects lanes, and zero otherwise.

## CudaDetection.py

CudaDetection is an attempt at converting our current LaneDetection code into code that utilizes the Cuda library. It contains the same functions as LaneDetection.py, but those functions use the Cuda library. Once the code is fully converted, we should be able to use this file as simply as we use the LaneDetection file.

## ADSDetection.py

ADSDetection is our legacy file. It was written for the 2022 competition and has not been edited much since then. It most likely will not work with the new Jetson Orin, since the Orin can run newer versions of ROS and OpenCV. This file is useful to see syntax for listening to multiple topics in ROS, and it is useful to see what we ran last year.