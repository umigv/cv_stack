#!/usr/bin/env python
# file for lane detection without perspective transform
# importing some useful packages
import rospy
import sys
from sensor_msgs.msg import Image, CameraInfo
import numpy as np
import cv2
import sys
from cv_bridge import CvBridge
import message_filters


# The hard work of the CV subteam of UMARV

class ADSDetection:
    
    def __init__(self, image, depth_image):
        # dilate the depth_image
        global depthMap
        kernel_size = 5
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        iterations = 4
        depthMap = self.dilate(depth_image, kernel, iterations)

        global gaussianBlur
        kernel_size = 3
        gaussianBlur = self.gaussian_blur(image, kernel_size)

        # color threshold
        minThreshold = 200
        maxThreshold = 255
        global thresholded
        thresholded = self.colorThreshold(gaussianBlur, minThreshold, maxThreshold)

        kernel_size = 2
        iterations = 11
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)
        thresholded = self.dilate(thresholded, kernel, iterations)
        

        global cropped
        # crop the image
        height = image.shape[0]
        width = image.shape[1]
        region_of_interest_vertices = [
            (int(width/2), int(5*height/6)),
            (0, height),
            (0, int(height/5)),
            (width, int(height/ 5)),
            (width, height),
        ]

        cropped = self.region_of_interest(thresholded, np.array([region_of_interest_vertices]))

        # grayscale
        cropped = self.grayscale(cropped)

        # # canny
        # minThreshold = 150
        # maxThreshold = 230
        # global edgeDetectedImage
        # edgeDetectedImage = self.cannyEdgeDetection(grayImage, minThreshold, maxThreshold)

        global maskedImage
        maskedImage = self.mask(depthMap, cropped.astype(np.uint8))

        # global count
        # global cur_image
        # if count > 100 and count % 30 == 0:
        #     # output_cropped = self.mask(cropped, depth_image.astype(np.uint8))
        #     path = f"./depth_image{cur_image}.png"
        #     worked1 = cv2.imwrite(path, depth_image)
        #     path = f"./cropped{cur_image}.png"
        #     worked2 = cv2.imwrite(path, cropped)
        #     rospy.loginfo(worked1)
        #     rospy.loginfo(worked2)
        #     rospy.loginfo("image written")
        #     cur_image += 1
        # count += 1

        # global dilatedImage
        # if we're using dilation

        # kernel_size = 5
        # kernel = np.ones((kernel_size, kernel_size), np.uint8)
        # iterations = 3
        # dilatedImage = cv2.morphologyEx(maskedImage, cv2.MORPH_OPEN, (kernel_size, kernel_size))
        # dilatedImage = self.dilate(dilatedImage, kernel, iterations)

        # no dilation
        # dilatedImage = maskedImage 

        # # image for display, not for use in pipeline
        # global displayImage
        # displayImage = dilatedImage
        # displayImage[displayImage > 0] = 255

    def colorThreshold(self, img, minThreshold, maxThreshold):

        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        # define range of white color in HLS
        lower_white = np.array([0,minThreshold,0], dtype=np.uint8)
        upper_white = np.array([255,maxThreshold,255], dtype=np.uint8)
        # Threshold the HSV image to get only white colors
        mask = cv2.inRange(hls, lower_white, upper_white)
        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(hls, hls, mask= mask)

        return res
    
    def region_of_interest(self,img, vertices):
        """
        Applies an image mask.

        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        """
        #defining a blank mask to start with
        mask = np.zeros_like(img)

        #defining a 3 channel or 1 channel color to fill the mask with
        #depending on the input image
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        #filling pixels inside the polygon defined by "vertices" with the fill color
        cv2.fillPoly(mask, np.int32([vertices]), ignore_mask_color)

        #returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image
    
    def cannyEdgeDetection(self, img, minThreshold, maxThreshold):
        return cv2.Canny(img, minThreshold, maxThreshold)

    def grayscale(self, img):
        """Applies the Grayscale transform
        This will return an image with only one color channel
        but NOTE: to see the returned image as grayscale
        you should call plt.imshow(gray, cmap='gray')"""
        # since the image will be in hls, we convert to bgr then gray
        image = cv2.cvtColor(img, cv2.COLOR_HLS2BGR)
        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return image

    def gaussian_blur(self, img, kernel_size):
        """Applies a Gaussian Noise kernel"""
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 200)
    
    def mask(self, depth_image, mask):
        return cv2.bitwise_and(depth_image, depth_image, mask=mask)
    
    def dilate(self, img, kernel, iterations):
        return cv2.dilate(img, kernel, iterations=iterations)

    def returnThresholdedImage(self):
        global thresholded

        # there's no hls to gray conversion, so have to go to bgr first
        # can also try extracting just the luminance column
        thresholded = cv2.cvtColor(thresholded, cv2.COLOR_HLS2BGR)
        thresholded = cv2.cvtColor(thresholded, cv2.COLOR_BGR2GRAY)

        return thresholded
    
    def returnDepthMap(self):
        global depthMap
        return depthMap

    def returnGaussianBlur(self):
        global gaussianBlur

        # there's no hls to gray conversion, so have to go to bgr first
        # can also try extracting just the luminance column
        gaussianBlur = cv2.cvtColor(gaussianBlur, cv2.COLOR_HLS2BGR)
        gaussianBlur = cv2.cvtColor(gaussianBlur, cv2.COLOR_BGR2GRAY)

        return gaussianBlur
    
    def returnCroppedImage(self):
        global cropped
        return cropped

    def returnEDImage(self):
        global edgeDetectedImage
        return edgeDetectedImage
    
    def returnMaskedImage(self):
        global maskedImage
        return maskedImage
    
    def returnDisplayImage(self):
        global displayImage
        return displayImage

camera_info_ros = CameraInfo()
transformed_ros = Image()

def callback(image, depth_image, camera_info):
    #rospy.loginfo("Converting perspective transformed img to edge detection image")
    bridge = CvBridge()
    # timestamp = image.header.stamp
    cv_image = bridge.imgmsg_to_cv2(image, desired_encoding='bgr8')
    depth_image = bridge.imgmsg_to_cv2(depth_image, desired_encoding='passthrough')
    # depth_image = cv2.dilate(depth_image, (5, 5), iterations=10)
    ads = ADSDetection(cv_image, depth_image)
    publish_transform(bridge, ads.returnCroppedImage())
    """
    we are publishing this because for some reason the depth.launch
    file is looking for a topic with this exact name, even though
    I feel that it should be able to find /zed2i/zed_node/depth/camera_info
    This is a workaround, we are not editing the camera info at all
    """
    publish_camera_info(camera_info)

def publish_transform(bridge, cv2_img):
    global transformed_ros
    transformed_ros = bridge.cv2_to_imgmsg(cv2_img)
    transformed_ros.header.frame_id = "zed_left_camera_optical_frame"

    # pub = rospy.Publisher(topic, Image, queue_size=1)
    # pub.publish(transformed_ros)
    # rospy.loginfo("Published transform")

def publish_camera_info(camera_info):
    global camera_info_ros
    camera_info_ros = camera_info

    # pub = rospy.Publisher(topic, CameraInfo, queue_size=1)
    # pub.publish(camera_info)
    # rospy.loginfo("Published camera info")


def listener():
    global camera_info_ros
    global transformed_ros
    global count
    global cur_image
    cur_image = 1
    count = 0
    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('ads_detection', anonymous=True)
    global POTHOLES_ENABLED
    msg = "Initialized Lane Detection: " + "No potholes" if not POTHOLES_ENABLED else "Potholes Enabled"
    rospy.loginfo(msg)

    image_sub = message_filters.Subscriber("/zed2i/zed_node/left/image_rect_gray", Image, queue_size = 1, buff_size=2**24)
    # image_sub = message_filters.Subscriber("/zed2i/zed_node/left/image_rect_color", Image)
    depth_map_sub = message_filters.Subscriber("/zed2i/zed_node/depth/depth_registered", Image, queue_size = 1, buff_size=2**24)

    camera_info_sub = message_filters.Subscriber("/zed2i/zed_node/depth/camera_info", CameraInfo, queue_size = 1, buff_size=2**24)
    ts = message_filters.TimeSynchronizer([image_sub, depth_map_sub, camera_info_sub], 10)
    ts.registerCallback(callback)

    rate = rospy.Rate(10)
    transform_pub = rospy.Publisher("/cv/laneMapping/left", Image, queue_size=1)
    camera_info_pub = rospy.Publisher("/cv/laneMapping/camera_info", CameraInfo, queue_size=1)

    log = rospy.get_param("log_info")

    while not rospy.is_shutdown():
        rate.sleep()
        time = rospy.Time.now()
        transformed_ros.header.stamp = time
        camera_info_ros.header.stamp = time
        transform_pub.publish(transformed_ros)
        camera_info_pub.publish(camera_info_ros)
        if log:
            rospy.loginfo("Published transform and camera info")
        

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


if __name__ == '__main__':
    try:
        argv = rospy.myargv(argv=sys.argv)
        global POTHOLES_ENABLED
        if (len(argv) == 2):
            POTHOLES_ENABLED = True
        else:
            POTHOLES_ENABLED = False
        listener()
    except rospy.ROSInterruptException:
        pass
