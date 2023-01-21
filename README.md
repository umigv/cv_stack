# cv_stack

## Synopsis

`cv_stack` contains the codebase for the Computer Vision subteam of UMARV. Most of the code was written in the 2021-2022 season.

## Motivation

This code contains various applications. It contains ROS nodes for lane detection, perspective transform, ROS image topic visualization, and more. It also contains the onboarding projects for the 2022/2023 season. This code was written with large help from the OpenCV python library.refer 

## Contributors

`cv_stack` is maintained by the CV Subteam in ARV.

# How To Run

Run Commands in the Google Drive, under 2022-2023/Computer Vision will contain most commands you might want to use. This will be an explanation of the most used commands

`roscore` can be run before all other commands, but is usually not necessary

`roslaunch zed_wrapper zed.launch` will run the zed ros node with the camera plugged in. If you want to run it with an svo file, add the `svo_file:="/path/to/file.svo"` argument.

`rosrun lane_mapping LaneDetection.py` will run the lane detection ros node. You can swap `LaneDetection.py` with `CudaDetection.py` or `ADSDetection.py` to run those python files. Add any argument afterwards to run with pothole detection enabled.

`rosrun visualize_topic visualizer.py` will visualize a topic with cv2.imshow, make sure to run `export DISPLAY=:0` first if you are running with ssh. If you get an error about a compressed image, use `rosrun visualize_topic visualize_compressed.py` instead.

`rostopic list` will list all of the topics currently available. `rostopic hz topic_name` will display a running counter of the rate that topic_name is being published.