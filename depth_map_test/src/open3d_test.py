import open3d
import cv2
import numpy as np
# from PIL import Image

FX = 708.4204711914062
FY = 708.4204711914062
CX = 545.5575561523438
CY = 333.53106689453125

def mess_with_depth_map(depth_numpy):
    # this mask should be lane detected image
    # mask = np.ones_like(depth_image) * 255
    
    # now the depth image only contains the lanes
    # masked_image = cv2.bitwise_and(depth_image, mask)
    # img_array = np.array(depth_image)
    
    # img_array = np.array(depth_image)
    # print(img_array)

    # converting to open3d image first
    image2 = open3d.geometry.Image(depth_numpy)
    open3d.visualization.draw_geometries([image2])

    # need to set the camera intrinsics in order to use the point cloud
    # intrins = open3d.camera.PinholeCameraIntrinsic()
    # intrins.set_intrinsics(img_array.shape[1], img_array.shape[0], FX, FY, CX, CY)
    
    # creating the point cloud from open3d image
    # point_cloud = open3d.geometry.PointCloud.create_from_depth_image(image2, intrins, depth_scale=100)

    # visualize open3d point cloud
    # point_cloud.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    # open3d.visualization.draw_geometries([point_cloud])

# image = Image.open('depth_test.png')

image  = np.genfromtxt('../depth_map.csv', delimiter=",") 
mess_with_depth_map(image)