# 提高对比度和亮度、将目标区域变为灰度图像

import cv2 as cv
import numpy as np
from image_merge import plt_pic


# 提高对比度和亮度
def contrast_brightness(path):
    img2 = cv.imread(path)

    if img2 is None:
        print("无法读取图片")
        return -1

    c = 0.7
    b = 20
    blank = np.zeros(img2.shape, img2.dtype)
    dst = cv.addWeighted(img2, c, blank, 1 - c, b)     # 提高对比度，即增加权重
    image_dict = {"original2": img2, "contrast_brightness_windows": dst}
    plt_pic(image_dict)


# 将脸部小部分变为灰度图像
def face_gray(path):
    lean = cv.imread(path)

    if lean is None:
        print("无法读取图片")
        return -1

    face = lean[200:400, 180:380]
    # 先转为灰度图,再转为RGB图像
    face = cv.cvtColor(face, cv.COLOR_BGR2GRAY)
    face = cv.cvtColor(face, cv.COLOR_GRAY2BGR)
    lean[200:400, 180:380] = face
    image_dict = {"original": lean}
    plt_pic(image_dict)


if __name__ == "__main__":
    image_path = r"./images/lena.jpg"
    # contrast_brightness(image_path)
    face_gray(image_path)
