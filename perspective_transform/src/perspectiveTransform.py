#!/usr/bin/env python
# license removed for brevity
# from tkinter import Image
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from math import radians, cos
import numpy as np
from rospy.numpy_msg import numpy_msg
# from rospy_tutorials.msg import Floats
import ros_numpy

import cv2


def getBirdView(image, cp):#TODO could be sped up by only doing math once
    if (cp.matrix is None):
        rows, columns = image.shape[:2]
        #rospy.loginfo(rows) #376 for non svo   720 for svo
        #rospy.loginfo(columns) #672 for non svo    1280 for svo
        if columns == 1280:
            columns = 1344
        if rows == 720:
            rows = 752
        min_angle = 0.0
        max_angle = cp.compute_max_angle()
        min_index = cp.compute_min_index(rows, max_angle)
        image = image[min_index:, :]
        rows = image.shape[0]

        src_quad = cp.src_quad(rows, columns)
        dst_quad = cp.dst_quad(rows, columns, min_angle, max_angle)
        return perspective(image, src_quad, dst_quad, cp)
    else:
        image = image[cp.minIndex:, :]
        return cv2.warpPerspective(image, cp.matrix, (cp.maxWidth, cp.maxHeight))



def perspective(image, src_quad, dst_quad, cp):
    bottomLeft, bottomRight, topLeft, topRight = dst_quad
    # solve for the new width
    widthA = topRight[0] - topLeft[0]
    widthB = bottomRight[0] - bottomLeft[0]
    maxWidth1 = max(widthA, widthB)
    # solve for the new height
    heightA = bottomLeft[1] - topLeft[1]
    heightB = bottomRight[1] - topRight[1]
    maxHeight1 = max(heightA, heightB)

    matrix1 = cv2.getPerspectiveTransform(src_quad, dst_quad)
    cp.matrix = matrix1
    cp.maxWidth = maxWidth1
    cp.maxHeight = maxHeight1
    return cv2.warpPerspective(image, matrix1, (maxWidth1, maxHeight1))


def rotate(image, angle, scale = 1.0):
    #https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
    height, width = image.shape[:2]
    cX, cY = (width//2, height//2)

    matrix = cv2.getRotationMatrix2D((cX, cY), -angle, scale)
    matrix_cos = np.abs(matrix[0, 0])
    matrix_sin = np.abs(matrix[0, 1])

    height_rotated = int((height*matrix_cos) + (width*matrix_sin))
    width_rotated = int((height*matrix_sin) + (width*matrix_cos))

    matrix[0, 2] += (width_rotated/2) - cX
    matrix[1, 2] += (height_rotated/2) - cY
    return cv2.warpAffine(image, matrix, (width_rotated, height_rotated))


class CameraProperties(object):
    # in an ideal world, computations would work until reaching the horizon (89.999...)
    # in the real world, the resulting distances get too big to handle
    # -- could go up to around 84.0, but to avoid lots of distortion stay a bit lower
    functional_limit = radians(77.0)
    def __init__(self, height, fov_vert, fov_horz, cameraTilt):
        '''height: the height above the ground, in any unit
        fov_vert: the vertical field of view, in degrees
        fov_horz: the horizontal field of view, in degrees
        cameraTilt: an acute angle measured from the ground, in degrees'''
        self.height = float(height)
        self.fov_vert = radians(float(fov_vert))
        self.fov_horz = radians(float(fov_horz))
        self.cameraTilt = radians(float(cameraTilt))
        self.bird_src_quad = None
        self.bird_dst_quad = None
        self.matrix = None
        self.maxHeight = None
        self.maxWidth = None
        self.minIndex = None

    def src_quad(self, rows, columns):
        '''This just finds the vertices of a rectangle that covers the entire standard image'''
        if self.bird_src_quad is None:
            # bottom left, bottom right, top left, top right
            self.bird_src_quad = np.array([[0, rows - 1], [columns - 1, rows - 1], [0, 0], [columns - 1, 0]], dtype = 'float32')
        return self.bird_src_quad

    def dst_quad(self, rows, columns, min_angle, max_angle):
        '''This finds 4 points such that dragging the standard image corners onto the points results in a birds eye view'''
        if self.bird_dst_quad is None:
            # fov_offset is the angle between the bottom of the vertical FOV and the ground's normal
            fov_offset = self.cameraTilt - self.fov_vert/2.0
            # bottom_over_top represents the ratio of the base lengths of the trapezoidal image that will be formed
            bottom_over_top = cos(max_angle + fov_offset)/cos(min_angle + fov_offset)
            # dimensional analysis
            bottom_width = columns*bottom_over_top
            # since bottom is thinner than top, create black spaces to keep it centered
            blackEdge_width = (columns - bottom_width)/2
            leftX = blackEdge_width
            rightX = leftX + bottom_width
            # bottom left, bottom right, top left, top right
            self.bird_dst_quad = np.array([[leftX, rows], [rightX, rows], [0, 0], [columns, 0]], dtype = 'float32')
        return self.bird_dst_quad

    def reset(self):
        self.bird_src_quad = None
        self.bird_dst_quad = None
        self.matrix = None
        self.maxHeight = None
        self.maxWidth = None
        self.minIndex = None

    def compute_min_index(self, rows, max_angle):
        self.minIndex = int(rows*(1.0 - max_angle/self.fov_vert))
        return self.minIndex
    def compute_max_angle(self):
        return min(CameraProperties.functional_limit - self.cameraTilt + self.fov_vert/2.0, self.fov_vert)



def callback_left(data):
    # rospy.loginfo("Converting zed left to perspective transform")
    bridge = CvBridge()
    timestamp = data.header.stamp
    cv_image = bridge.imgmsg_to_cv2(data, desired_encoding='bgra8')
    transformed_image = getBirdView(cv_image, ZED)
    publish_transform(bridge, "/cv/perspective/left", transformed_image, timestamp)
    # publish_dst("/cv/perspective/dst_quad", ZED.bird_dst_quad, timestamp)
    # rospy.loginfo(ZED.bird_dst_quad)
    msg = ros_numpy.msgify(Image, ZED.bird_dst_quad, encoding='32FC1')
    publish_dst("/cv/perspective/dst_quad", msg, timestamp)

'''
def callback_right(data):
    # rospy.loginfo("Converting zed right to perspective transform")
    bridge = CvBridge()
    timestamp = data.header.stamp
    cv_image = bridge.imgmsg_to_cv2(data, desired_encoding='bgra8')
    transformed_image = getBirdView(cv_image, ZED)
    publish_transform(bridge, "/cv/perspective/right", transformed_image, timestamp)
'''

def publish_transform(bridge, topic, cv2_img, timestamp):
    transformed_ros = bridge.cv2_to_imgmsg(cv2_img, encoding='bgra8')
    transformed_ros.header.stamp = timestamp
    pub = rospy.Publisher(topic, Image, queue_size=1)
    pub.publish(transformed_ros)
    rospy.loginfo("Published transform")

def publish_dst(topic, vertices, timestamp):
    vertices.header.stamp = timestamp
    pub = rospy.Publisher(topic, Image, queue_size=1)
    pub.publish(vertices)
    rospy.loginfo("Published dst_quad")

    
def listener():
    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('perspective_transform')

    global LEFT_TOPIC#, RIGHT_TOPIC
    LEFT_TOPIC = "/zed/zed_node/left/image_rect_gray"
    '''
    RIGHT_TOPIC = "/zed/zed_node/right/image_rect_gray"
    '''
    rospy.loginfo("Perspective Transform Initialized. Listening to the following messages: ")
    rospy.loginfo(LEFT_TOPIC)
    '''
    rospy.loginfo(RIGHT_TOPIC)
    '''

    rospy.Subscriber(LEFT_TOPIC, Image, callback_left, queue_size = 1, buff_size=2**24)
    '''
    rospy.Subscriber(RIGHT_TOPIC, Image, callback_right)
    '''
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


def init():
	# iniailize
    global ZED
    ZED = CameraProperties(43.18, 96.0, 54.0, 90.0) #TODO get accurate first parameter, which is height


if __name__ == '__main__':
    init()
    try:
        listener()
    except rospy.ROSInterruptException:
        pass
