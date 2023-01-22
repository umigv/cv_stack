# testing_scripts

## Synopsis

`testing_scripts` contains python scripts that run our lane detection code without using ROS. The purpose of these are so that they can be downloaded to an individual's laptop and run locally, or just run on the Jetson without ROS. All of these scripts will take a video as a command line argument. For example, `local_lane_detection.py video1.mp4` will run our lane detection algorithm.