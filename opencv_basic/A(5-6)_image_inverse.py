# 图像反转

import cv2 as cv
import numpy as np
from image_merge import plt_pic


# 自编函数进行图像反转
def access_pixels_user(path, show=True):
    image = cv.imread(path)

    if image is None:
        print("无法读取图片")
        return -1

    high, width, channel = image.shape
    image_type = image.dtype
    dst = np.zeros([high, width, channel], image_type)
    for row in range(high):
        for col in range(width):
            for c in range(channel):
                dst[row, col, c] = 255 - image[row, col, c]
    if show:
        image_dict = {"original": image, "图像反转": dst}
        plt_pic(image_dict)


# 运用open-cv自带函数进行图像反转
def access_pixels_inverse(path, show=True):
    image = cv.imread(path)

    if image is None:
        print("无法读取图片")
        return -1

    dst = cv.bitwise_not(image)      # 调用open-cv自带的“非”运算，即反转图像
    if show:
        image_dict = {"original": image, "图像反转": dst}
        plt_pic(image_dict)


def time_compute(image_path):
    t1 = cv.getTickCount()   # 用于统计函数运行的时间
    access_pixels_user(image_path, show=False)
    t2 = cv.getTickCount()
    time = (t2 - t1) / cv.getTickFrequency()
    # print("user_time:%s ms"%(time*1000))
    print("user_time:%s s" % time)

    t3 = cv.getTickCount()
    access_pixels_inverse(image_path, show=False)
    t4 = cv.getTickCount()
    time = (t4 - t3) / cv.getTickFrequency()
    # print("cv_time:%s ms"%(time*1000))
    print("open-cv_time:%s s" % time)

if __name__ == "__main__":
    image_path = r"./images/01.jpg"
    access_pixels_user(image_path)
    # access_pixels_inverse(image_path)
    # time_compute(image_path)
