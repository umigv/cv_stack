#!/usr/bin/env python
import open3d
import rospy
import numpy as np
import cv2

from sensor_msgs.msg import Image, PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2

import rospy
from std_msgs.msg import String, Header

import sys
from cv_bridge import CvBridge
from rospy.numpy_msg import numpy_msg
import message_filters

# camera params, should not change ever
FX = 708.4204711914062
FY = 708.4204711914062
CX = 545.5575561523438
CY = 333.53106689453125

FIELDS_XYZ = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
]

def mess_with_depth_map(cv_image, depth_image):
    # # this mask should be lane detected image
    # mask = np.ones_like(depth_image) * 255

    # # now the depth image only contains the lanes
    # masked_image = cv2.bitwise_and(depth_image, mask)
    print(depth_image)
    img_array = np.array(depth_image.astype(np.float32))

    # converting to open3d image first
    image2 = open3d.geometry.Image(img_array)

    # need to set the camera intrinsics in order to use the point cloud
    intrins = open3d.camera.PinholeCameraIntrinsic(open3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    # intrins.set_intrinsics(img_array.shape[1], img_array.shape[0], FX, FY, CX, CY)

    # creating the point cloud from open3d image
    point_cloud = open3d.geometry.PointCloud.create_from_depth_image(image2, intrins, depth_scale=100)

    # visualize open3d point cloud
    point_cloud.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    open3d.visualization.draw_geometries([image2])

    # converting to ROS point cloud from open3d point cloud
    ros_point_cloud = convertCloudFromOpen3dToRos(point_cloud)
    publish_point_cloud(ros_point_cloud)

def convertCloudFromOpen3dToRos(open3d_cloud, frame_id="odom"):
    # Set "header"
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id

    # Set "fields" and "cloud_data"
    points=np.asarray(open3d_cloud.points)
    if not open3d_cloud.colors: # XYZ only
        fields=FIELDS_XYZ
        cloud_data=points
    else: # XYZ + RGB
        fields=FIELDS_XYZRGB
        # -- Change rgb color from "three float" to "one 24-byte int"
        # 0x00FFFFFF is white, 0x00000000 is black.
        colors = np.floor(np.asarray(open3d_cloud.colors)*255) # nx3 matrix
        colors = colors[:,0] * BIT_MOVE_16 +colors[:,1] * BIT_MOVE_8 + colors[:,2]  
        cloud_data=np.c_[points, colors]
    
    # create ros_cloud
    return pc2.create_cloud(header, fields, cloud_data)

def depth_to_point_cloud(img):
    # shape (621, 1104)
    data = []
    # iterate through each pixel :(
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            depth_val = img[x][y]
            if depth_val > 0:
                # Compute the x, y, and z coordinates of each point in the point cloud
                X = (x - CX) * depth_val / FX
                Y = (y - CY) * depth_val / FY
                Z = depth_val
                data.append((X, Y, Z))

    publish_point_cloud(np.array(data).astype(np.uint8).ravel().tolist())

def publish_point_cloud(point_cloud):

    topic = '/cv/laneMapping/point_cloud'
    pub = rospy.Publisher(topic, PointCloud2, queue_size=3)

    pub.publish(point_cloud)
    rospy.loginfo("Published point cloud")


def callback(data, depth_map):
    #rospy.loginfo("Converting perspective transformed img to edge detection image")
    bridge = CvBridge()
    timestamp = data.header.stamp
    cv_image = bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
    depth_image = bridge.imgmsg_to_cv2(depth_map, desired_encoding='passthrough')
    mess_with_depth_map(cv_image, depth_image)
    # publish_transform(bridge, "/cv/laneMapping/left", ads.returnCroppedImage(), timestamp)


def listener():
    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('depth_map_testing', anonymous=True)
    rospy.loginfo("initialized depth_map_testing")

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
    #capture the Interrupt signals
    except rospy.ROSInterruptException:
        pass