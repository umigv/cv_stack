import numpy as np
import cv2


def main():
    num = 3
    image = cv2.imread(f"./depth_maps/cropped{num}.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    depth_image = cv2.imread(f"./depth_maps/depth_image{num}.png")
    depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY)
    norm_depth_image = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imshow("window", norm_depth_image)
    cv2.waitKey(0)

    kernel_size = 5
    # make it dilate as an ellipse and not a rectangle (not that important)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    iterations = 3
    dilated_map = dilate(depth_image, kernel, iterations)
    norm_depth_image = cv2.normalize(dilated_map, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imshow("window", norm_depth_image)
    cv2.waitKey(0)
    
    # masked_image = mask(dilated_map, image.astype(np.uint8))
    # norm_depth_image = cv2.normalize(masked_image, None, 0, 255, cv2.NORM_MINMAX)
    # cv2.imshow("window", norm_depth_image)
    # cv2.waitKey(0)

    # kernel = np.array([
    #     [0, 0, 1, 0, 0],
    #     [1, 0.5, 1, 2, 2],
    #     [1, 0.5, 1, 2, 2],
    #     [1, 0.5, 1, 2, 2],
    #     [0, 0, 1, 0, 0]], dtype=np.uint8)
    # iterations = 3
    # image = dilate(depth_image, kernel, iterations)
    # cv2.imshow("window", image)
    # cv2.waitKey(0)

    # image = cv2.morphologyEx(image, cv2.MORPH_OPEN, (kernel_size,kernel_size))
    # image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    #image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)


def dilate(img, kernel, iterations):
    return cv2.dilate(img, kernel, iterations=iterations)

def mask(depth_image, mask):
    return cv2.bitwise_and(depth_image, depth_image, mask=mask)

if __name__ == '__main__':
    main()
