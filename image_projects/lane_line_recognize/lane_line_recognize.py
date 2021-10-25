# 求车尾方向 距离图像中心点 最近的 车道线中心 的线段方程，即直线方程 + 端点坐标
# 先验条件：
# 1、车道线亮色（黄色），可以转化为灰度图处理
# 2、车道线和图像的大致夹角已知（正负15度），可自行测量作为已知量输入
# 3、每个像素长宽方向均代表 1.2 cm，供参考
# 4、所有车道线均为正交，即满足互相平行 或者 互相垂直 的关系

# 参考思路：先将车道线旋转到与坐标系平行，转灰度图，用6个像素（约7.5cm 一半车道线宽）的滑动窗口计算梯度值得到车道线中心点

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from image_merge import plt_pic


def car(img, image_dict):
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lower_hsv = np.array([0, 43, 46])  # h,s,v的下范围，可查“HSV基本颜色分量范围”
    upper_hsv = np.array([10, 255, 255])  # h,s,v的上范围，此处提取的为红色
    mask = cv.inRange(img_hsv, lowerb=lower_hsv, upperb=upper_hsv)  # 提取设置范围内的颜色
    dst = cv.bitwise_and(img, img, mask=mask)  # 先对img求和，然后再用mask提取想要的区域
    image_dict['dst'] = dst

    return image_dict


def lane_line_recognize(img_path):
    img = cv.imread(img_path)

    if img is None:
        print("无法读取图片")
        return -1

    # plt.hist(img.ravel(), 256, [0, 256])
    # plt.show()
    median = cv.medianBlur(img, 15)
    gray = cv.cvtColor(median, cv.COLOR_BGR2GRAY)
    # median = cv.medianBlur(gray, 5)
    binary = cv.threshold(gray, 70, 255, cv.THRESH_BINARY_INV)[1]
    contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # 查找轮廓
    cv.drawContours(img, contours, -1, (0, 0, 255), 3)

    # guss = cv.GaussianBlur(img, (5, 5),  0, 0)
    # guss = cv.blur(img, (3, 3))
    # guss = cv.medianBlur(img, 15)
    # guss = cv.bilateralFilter(img, 0, 100, 15)
    # guss = cv.pyrMeanShiftFiltering(img, 10, 50)
    # grad = cv.Canny(guss, 30, 100)

    # 自定义函数显示图片
    image_dict = {"original": img, "binary": binary, "median": median}
    # image_dict = {"original": img, "binary": binary, "guss": guss, "grad": grad}
    # image_dict = car(img, image_dict)
    plt_pic(image_dict)


if __name__ == "__main__":
    path = "HUANLEGU-20210610-13028801887-105835-1-B9829-src.jpeg"
    lane_line_recognize(path)

