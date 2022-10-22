#!/usr/bin/env python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def findLines(image):
    def region_of_interest(img, vertices):
        mask = np.zeros_like(img)
        match_mask_color = 255 # <-- This line altered for grayscale.
        
        cv2.fillPoly(mask, vertices, match_mask_color)
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    def draw_lines(img, lines, color=[255, 0, 0], thickness=3):
        # If there are no lines to draw, exit.
        if lines is None:
            return
        # Make a copy of the original image.
        img = np.copy(img)
        # Create a blank image that matches the original in size.
        line_img = np.zeros(
            (
                img.shape[0],
                img.shape[1],
                img.shape[2]
            ),
            dtype=np.uint8,
        )
        # Loop over all lines and draw them on the blank image.
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
        # Merge the image with the lines onto the original.
        img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
        # Return the modified image.
        return img



    region_of_interest_vertices = [
        (0, image.shape[0]),
        (image.shape[1] / 2, image.shape[0] / 2),
        (image.shape[1], image.shape[0]),
    ]
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    cannyed_image = cv2.Canny(gray_image, 100, 200)
    # Moved the cropping operation to the end of the pipeline.
    cropped_image = region_of_interest(
        cannyed_image,
        np.array([region_of_interest_vertices], np.int32)
    )
    lines = cv2.HoughLinesP(
        cropped_image,
        rho=6,
        theta=np.pi / 60,
        threshold=160,
        lines=np.array([]),
        minLineLength=40,
        maxLineGap=25
    )
    image = draw_lines(image, lines)
    rospy.loginfo(type(image))
    return image
    

def publish_transform(bridge, topic, cv2_img, timestamp):
    transformed_ros = bridge.cv2_to_imgmsg(cv2_img, encoding='bgra8')
    transformed_ros.header.stamp = timestamp
    pub = rospy.Publisher(topic, Image, queue_size=1)
    pub.publish(transformed_ros)
    rospy.loginfo("Published example lane detection")

def callback(data):
    bridge = CvBridge()
    timestamp = data.header.stamp
    cv_image = bridge.imgmsg_to_cv2(data, desired_encoding='bgra8')
    img = findLines(cv_image)
    publish_transform(bridge, "/cv/onboarding", img, timestamp)

    
def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('listener', anonymous=True)

    rospy.Subscriber("/zed/zed_node/left/image_rect_gray", Image, callback, queue_size = 1, buff_size=2**24)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()