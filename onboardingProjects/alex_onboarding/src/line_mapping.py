#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
import numpy as np
import cv2
from cv_bridge import CvBridge
import math


def region_of_interest(img, vertices):
    # Define a blank matrix that matches the image height/width.
    mask = np.zeros_like(img)

    # Create a match color
    match_mask_color = 255

    # Fill inside the polygon
    cv2.fillPoly(mask, vertices, match_mask_color)

    # Returning the image only where mask pixels match
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0, 255], thickness=3):
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
        x1, y1, x2, y2 = line
        cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
    # Merge the image with the lines onto the original.
    img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
    # Return the modified image.
    return img

def pipeline(image):
    # reading in an image
    # plt.imshow(image)

    # Crop it
    width = image.shape[1]
    height = image.shape[0]
    region_of_interest_vertices = [
        (0, height),
        (width / 2, height / 2),
        (width, height),
    ]

    # image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print('normal image')
    print(image.shape)
    canneyed_image = cv2.Canny(gray_image, 20, 100)

    cropped_image = region_of_interest(
        canneyed_image, np.int32([region_of_interest_vertices])
    )
    # out = cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2BGRA)
    # print(out)

    # return out
    # canny edge detection

    lines = cv2.HoughLinesP(
        cropped_image,
        rho=6,
        theta=np.pi / 60,
        threshold=160,
        lines=np.array([]),
        minLineLength=40,
        maxLineGap=25
    )
    

    # Filter the lines
    # if lines is None:
    #     return out
    left_line_x = []
    left_line_y = []
    right_line_x = []
    right_line_y = []
    for line in lines:
        for a in line:
            x1 = line[0][0]
            y1 = line[0][1]
            x2 = line[0][2]
            y2 = line[0][3]

            slope = (y2 - y1) / (x2 - x1) # <-- Calculating the slope.
            if math.fabs(slope) < 0.5: # <-- Only consider extreme slope
                continue
            if slope <= 0: # <-- If the slope is negative, left group.
                left_line_x.extend([x1, x2])
                left_line_y.extend([y1, y2])
            else: # <-- Otherwise, right group.
                right_line_x.extend([x1, x2])
                right_line_y.extend([y1, y2])

    # Map lines to properly span the horizon
    min_y = int(image.shape[0] * (3 / 5)) # <-- Just below the horizon
    max_y = int(image.shape[0]) # <-- The bottom of the image

    poly_left = np.poly1d(np.polyfit(
        x=left_line_y,
        y=left_line_x,
        deg=1
    ))
    left_x_start = int(poly_left(max_y))
    left_x_end = int(poly_left(min_y))
    # print(right_line_y)
    poly_right = np.poly1d(np.polyfit(
        x=right_line_y,
        y=right_line_x,
        deg=1
    ))
    right_x_start = int(poly_right(max_y))
    right_x_end = int(poly_right(min_y))

    line_image = draw_lines(image, 
        [
            (left_x_start, max_y, left_x_end, min_y),
            (right_x_start, max_y, right_x_end, min_y),
        ]
    )

    # print(lines)


    # plt.figure()
    # plt.imshow(line_image)
    # plt.show()
    return line_image


def publish_transform(bridge, topic, image, timestamp):
    rosimage = bridge.cv2_to_imgmsg(image, encoding='bgra8')
    rosimage.header.stamp = timestamp
    pub = rospy.Publisher(topic, Image, queue_size=1)
    pub.publish(rosimage)
    rospy.loginfo("Published image of lane mapping")

def callback(data):
    rospy.loginfo(rospy.get_caller_id() + " heard data from ZED camera")
    #rospy.loginfo("Converting perspective transformed img to edge detection image")
    bridge = CvBridge()
    timestamp = data.header.stamp
    cv_image = bridge.imgmsg_to_cv2(data, desired_encoding='bgra8')
    modified = pipeline(cv_image)
    publish_transform(bridge, '/cv/alex_onboarding', modified, timestamp)
    # rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)

def listener():
    rospy.init_node('lanemapper', anonymous=True)
    # rospy.Subscriber('chatter', String, callback)
    rospy.Subscriber("/zed/zed_node/left/image_rect_gray", Image, callback, queue_size = 1, buff_size=2**24)
    
    rospy.loginfo(rospy.get_caller_id() + " starting line-mapping.py")
    rospy.spin()

if __name__ == "__main__":
    listener()