# 滤波算法有：均值滤波、中值滤波、高斯模糊、自定义滤波,形态学滤波

import cv2 as cv
import numpy as np
from image_merge import plt_pic


# 均值滤波；中值滤波；高斯模糊；高斯双边；均值迁移
def blur_demo(path):
    lean = cv.imread(path)

    if lean is None:
        print("无法读取图片")
        return -1

    dst_average = cv.blur(lean, (5, 5))
    dst_median = cv.medianBlur(lean, 15)
    dst_Gaussian = cv.GaussianBlur(lean, (0, 0), 15)
    dst_bilateral = cv.bilateralFilter(lean, 0, 100, 15)
    dst_pyrMeanShift = cv.pyrMeanShiftFiltering(lean, 10, 50)
    image_dict = {"original": lean, "均值滤波": dst_average, "中值滤波": dst_median, 
                  "高斯模糊": dst_Gaussian, "高斯双边": dst_bilateral, "均值迁移": dst_pyrMeanShift}
    plt_pic(image_dict)


# 自定义滤波函数
def custom_blur(path):
    lean = cv.imread(path)

    if lean is None:
        print("无法读取图片")
        return -1

    kernel = (np.ones([5, 5], np.float32)) / 25    # 滤波
    kernel1 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)    # 锐化
    dst = cv.filter2D(lean, -1, kernel=kernel1)     # 自建滤波函数
    image_dict = {"original": lean, "自定义滤波": dst}
    plt_pic(image_dict)


# 溢出拨回函数，防止处理图像时，数值超过255或小于0
def clamp(pv):
    if pv > 255:
        return 255
    elif pv < 0:
        return 0
    else:
        return pv


# 自定义高斯滤波函数
def custom_gaussian_noise(path):  #感觉不对
    lean = cv.imread(path)

    if lean is None:
        print("无法读取图片")
        return -1

    dst = lean.copy()
    h, w, c = lean.shape
    for row in range(h):
        for col in range(w):
            s = np.random.normal(0, 20, 3)   # 创建一个高斯分布
            b = lean[row, col, 0]
            g = lean[row, col, 1]
            r = lean[row, col, 2]
            dst[row, col, 0] = clamp(b + s[0])
            dst[row, col, 1] = clamp(g + s[1])
            dst[row, col, 2] = clamp(r + s[2])
    image_dict = {"original": lean, "自定义高斯滤波": dst}
    plt_pic(image_dict)


if __name__ == "__main__":
    image_path = r"./images/lena.jpg"
    blur_demo(image_path)
    # custom_blur(image_path)
    # custom_gaussian_noise(image_path)
