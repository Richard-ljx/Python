# 色彩空间转换、利用HSV色彩空间提取目标区域、分离RGB图像

import cv2 as cv
import numpy as np
from image_merge import plt_pic


# 将RGB图像转换为几个常用的色彩空间图片
def color_space_demo(path):
    image = cv.imread(path)

    if image is None:
        print("无法读取图片")
        return -1

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)       # 转为灰度图像
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)         # 转入hsv色彩空间的图像，对色度(颜色)不敏感，具体可查百度百科
    yuv = cv.cvtColor(image, cv.COLOR_BGR2YUV)         # 转入yuv色彩空间的图像，对亮度值不敏感，具体可查百度百科
    ycrcb = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)     # 转入ycrcb色彩空间的图像，是yuv的改进版，可用于区分皮肤与背景
    image_dict = {"original": image, "gray": gray, "hsv": hsv, "yuv": yuv, "ycrcb": ycrcb}
    plt_pic(image_dict)



def extrace_object_demo():
    capture = cv.VideoCapture(0)     # 0表示电脑自带的摄像头

    if not capture.isOpened():
        print("摄像头打开失败")
        return -1

    while True:
        ret, frame = capture.read()
        frame = cv.flip(frame, 1)
        if not ret:
            break
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)   # 转换为hsv色彩空间，下面将利用其特性区分物体
        lower_hsv = np.array([0, 0, 0])              # h,s,v的下范围，可查“HSV基本颜色分量范围”
        upper_hsv = np.array([180, 255, 46])         # h,s,v的上范围，此处提取的为黑色
        # inRange会将位于两个区域间的值置为255，位于区间外的值置为0
        mask = cv.inRange(hsv, lowerb=lower_hsv, upperb=upper_hsv)    # 提取设置范围内的颜色
        dst = cv.bitwise_and(frame, frame, mask=mask)     # 先对frame求和，然后再用mask提取想要的区域
        cv.imshow("video", frame)
        cv.imshow("mask", dst)
        c = cv.waitKey(50)
        if c == 27:
            break


# 分离图像分别为b,g,r单通道图像
def split_bgr_show(path):
    image = cv.imread(path)

    if image is None:
        print("无法读取图片")
        return -1

    image_ori = image.copy()
    # 第一种方法，此方法显示为灰度图像
    b, g, r = cv.split(image)
    print(b.shape)
    src = cv.merge([b, g, r])
    # 第二种方法,此方法显示为彩色图像
    image[:, :, 2] = 0
    image[:, :, 1] = 0
    image_dict = {"original": image_ori, "b": b, "g": g, "r": r, "src": src, "blue": image}
    plt_pic(image_dict)


if __name__ == "__main__":
    image_path = r"./images/02.jpg"
    # color_space_demo(image_path)
    # split_bgr_show(image_path)

    extrace_object_demo()