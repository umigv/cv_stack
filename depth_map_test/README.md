# depth_map_test

## Synopsis

`depth_map_testing` is a ROS package that will be used for testing with the depth map from ZED and seeing if we can access Zed camera feed without using the Zed ROS wrapper. `sdk_depth_map_to_ros.py` will attempt to publish an image that is the depth map from the ZED SDK, instead of from the zed_ros_wrapper. `depth_map_testing.py` will access both the left image and the depth map from ROS for further testing.