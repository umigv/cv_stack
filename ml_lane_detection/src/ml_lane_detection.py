#!/usr/bin/env python
# file for lane detection without perspective transform
# importing some useful packages
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import rospy
import sys
from sensor_msgs.msg import Image
import numpy as np
import cv2
# import cupy as cp
import math
import sys
from cv_bridge import CvBridge
from rospy.numpy_msg import numpy_msg
import message_filters
from geometry_msgs.msg import Pose
import torch
import torch.nn as nn
import torchvision.transforms as transforms


#Define the model architecture
class lane_detection_model(nn.Module):

	def __init__(self):
		super(lane_detection_model, self).__init__()
		self.model = nn.Sequential(
			nn.Conv2d( in_channels=3 , out_channels=20 , kernel_size=15 , padding=7 , stride=1 ),
			nn.BatchNorm2d(20),
			nn.LeakyReLU(),
			nn.Conv2d( in_channels=20 , out_channels=10 , kernel_size=15 , padding=7 , stride=1 ),
			nn.BatchNorm2d(10),
			nn.LeakyReLU(),
			nn.Conv2d( in_channels=10 , out_channels=2 , kernel_size=15 , padding=7 , stride=1 ),
		)

	def forward(self, input):
		output = self.model(input)
		return output

class MLDetection:

	def __init__(self, image):
		# if "var" in globals():
		if torch.cuda.is_available():
			rospy.loginfo("Using the GPU! :)")
		else:
			rospy.loginfo("Using the CPU! :'(")
		self.device = torch.device("cuda")
		

		#Initialize the model
		self.ml_lane_detector = lane_detection_model().to(self.device)
		model_weight_path = "/home/umarv/catkin_ws/src/cv_stack/ml_lane_detection/model_weights.pth"
		# self.ml_lane_detector = torch.load(model_weight_path, map_location=torch.device('cpu'))

		self.ml_lane_detector.load_state_dict(torch.load(model_weight_path, map_location=self.device), strict=False) 
		rospy.loginfo("model initialized")
		tensor_transform = transforms.ToTensor()
		downscale_transform = transforms.Resize((128, 128))
		upscale_transform = transforms.Resize((621, 1104))

		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = tensor_transform(image)
		image = downscale_transform(image)
		image = image.to(self.device)
		image = image.unsqueeze(dim=0)
		# rospy.loginfo(f"{image.shape=}")
	
		lane_mask = self.ml_lane_detector(image)
		# lane_mask = lane_mask[0,1,:,:]
		# rospy.loginfo(lane_mask.shape)
		# lane_mask = upscale_transform(lane_mask)

		soft = nn.Softmax(dim=1)
		soft_output = soft(lane_mask)

		#Get the output of the ones class
		ones_output = soft_output[0,1,:,:]

		#Thresholded output
		output = torch.zeros(ones_output.shape, device=self.device)
		self.output_threshold = 0.3
		output[ones_output > self.output_threshold] = 1

		lane_mask = output
		lane_mask = lane_mask.detach().cpu().numpy()
		self.output = lane_mask
		rospy.loginfo("model run")

	def returnOutput(self):
		return self.output


def callback(image, depth_image):
	#rospy.loginfo("Converting perspective transformed img to edge detection image")
	bridge = CvBridge()
	timestamp = image.header.stamp
	cv_image = bridge.imgmsg_to_cv2(image, desired_encoding='bgr8')
	depth_image = bridge.imgmsg_to_cv2(depth_image, desired_encoding='passthrough')
	lanes = MLDetection(cv_image)
	publish_transform(bridge, "/cv/ml/left", lanes.returnOutput(), timestamp)

def publish_transform(bridge, topic, cv2_img, timestamp):
	transformed_ros = bridge.cv2_to_imgmsg(cv2_img)
	transformed_ros.header.stamp = timestamp
	pub = rospy.Publisher(topic, Image, queue_size=1)
	pub.publish(transformed_ros)
	rospy.loginfo("Published transform")

def listener():
	# In ROS, nodes are uniquely named. If two nodes with the same
	# name are launched, the previous one is kicked off. The
	# anonymous=True flag means that rospy will choose a unique
	# name for our 'listener' node so that multiple listeners can
	# run simultaneously.
	rospy.init_node('ml_lane_detection', anonymous=True)
	global POTHOLES_ENABLED
	msg = "Initialized ML Lane Detection"
	rospy.loginfo(msg)

	image_sub = message_filters.Subscriber("/zed/zed_node/left/image_rect_gray", Image, queue_size = 1, buff_size=2**24)
	# image_sub = message_filters.Subscriber("/zed/zed_node/left/image_rect_color", Image)
	depth_map_sub = message_filters.Subscriber("/zed/zed_node/depth/depth_registered", Image, queue_size = 1, buff_size=2**24)
	
	ts = message_filters.TimeSynchronizer([image_sub, depth_map_sub], 10)
	ts.registerCallback(callback)

	# spin() simply keeps python from exiting until this node is stopped
	rospy.spin()


if __name__ == '__main__':
	try:
		listener()
	except rospy.ROSInterruptException:
		pass
