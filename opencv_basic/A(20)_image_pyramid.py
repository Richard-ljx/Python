# PyrDown降采样：高斯金字塔

import cv2 as cv
from image_merge import plt_pic


# 高斯降采样
def pyramid_down_demo(image, image_dict):
    level = 3
    temp = image.copy()
    pyramid_images = []
    for i in range(level):
        dst = cv.pyrDown(temp)
        pyramid_images.append(dst)
        image_dict["pyramid_down"+str(i)] = dst
        temp = dst.copy()
    return pyramid_images, image_dict


# 高斯升采样
def laplian_demo(path):
    image = cv.imread(path)

    if image is None:
        print("无法读取图片")
        return -1

    image_dict = {}
    pyramid_images, image_dict = pyramid_down_demo(image, image_dict)
    level = len(pyramid_images)
    for i in range(level-1, -1, -1):
        if i < 1:
            expand = cv.pyrUp(pyramid_images[i], dstsize=image.shape[:2])
            lpls = cv.subtract(image, expand)
            image_dict["lpls_down_" + str(i)] = lpls
        else:
            expand = cv.pyrUp(pyramid_images[i], dstsize=pyramid_images[i-1].shape[:2])    # 还原
            lpls = cv.subtract(pyramid_images[i-1], expand)
            image_dict["lpls_down_" + str(i)] = lpls
    plt_pic(image_dict)


if __name__ == "__main__":
    image_path = r"./images/lena.jpg"
    laplian_demo(image_path)
