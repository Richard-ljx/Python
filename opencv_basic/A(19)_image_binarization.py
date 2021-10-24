# 图像二值化：普通二值化、局部二值化、自定义二值化、超大图像二值化

import cv2 as cv
import numpy as np
from image_merge import plt_pic

# 自定义阈值二值化
def customer_threshold_demo(path):
    image = cv.imread(path)

    if image is None:
        print("无法读取图片")
        return -1

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    m = np.reshape(gray, [1, w*h])
    mean = m.sum()/(w*h)
    ret, binary = cv.threshold(gray, mean, 255, cv.THRESH_BINARY)   # mean为阈值
    image_dict = {"original": image, "自定义二值化": binary}
    plt_pic(image_dict)


# 自动寻找阈值二值化
def threshold_demo(path):
    image = cv.imread(path)

    if image is None:
        print("无法读取图片")
        return -1

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_TRIANGLE)     # cv.THRESH_OTSU表示自动寻转阈值
    # 上一行最后，这里的cv.THRESH_TRIANGLE(最适用于单个波峰，最开始用于医学分割细胞)也可以换成cv.THRESH_OTSU(双波峰)
    print("threshold value:%s" % ret)
    image_dict = {"original": image, "普通二值化": binary}
    plt_pic(image_dict)


# 自适应阈值算法(局部二值化)
def local_threshold_demo(path):
    image = cv.imread(path)

    if image is None:
        print("无法读取图片")
        return -1

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # 自适应方法有 ADAPTIVE_THRESH_MEAN_C(均值) 或 ADAPTIVE_THRESH_GAUSSIAN_C(高斯加权)
    # 25：分割计算的区域大小；  10:常数，每个区域计算出的阈值的基础上在减去这个常数作为这个区域的最终阈值，可以为负数
    binary = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 25, 10)
    # 比块内的均值大10，就变成白色，其他是黑色
    image_dict = {"original": image, "局部二值化": binary}
    plt_pic(image_dict)


# 超大图像自适应二值化
def big_image_threshold_demo(path):
    image = cv.imread(path)

    if image is None:
        print("无法读取图片")
        return -1

    print(image.shape)
    ch = 256
    cw = 256
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    h, w = image.shape[:2]
    for row in range(0, h, ch):
        for col in range(0, w, cw):
            roi = gray[row:row+ch, col:col+cw]
            dst = cv.adaptiveThreshold(roi, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 127, 20)
            gray[row:row+ch, col:col+cw] = dst
            print(np.std(dst), np.mean(dst))
    cv.imwrite("big_image_threshold_demo.png", gray)
    image_dict = {"original": image, "超大图像二值化": gray}
    plt_pic(image_dict)


if __name__ == "__main__":
    image_path = r"./images/lena.jpg"
    big_image_path = r"./images/big_image.jpg"
    # customer_threshold_demo(image_path)
    # threshold_demo(image_path)
    # local_threshold_demo(image_path)
    big_image_threshold_demo(big_image_path)
