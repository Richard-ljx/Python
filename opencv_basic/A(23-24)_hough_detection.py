# 现在霍夫检测可以用于任何曲线的检测，只要有曲线的公式参数

import cv2 as cv
import numpy as np
from image_merge import plt_pic


# 霍夫直线检测
def line_detection(path):
    image = cv.imread(path)

    if image is None:
        print("无法读取图片")
        return -1

    image_ori = image.copy()
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 50, 150, apertureSize=3)   # 上下限改成25,120，可以降低直线识别的门槛
    # cv2.HoughLines（image,rho,theta,threshold）,image 是输入图像;rho 为以像素为单位的距离 r 的精度。一般情况下，使用的精度是 1;
    # theta 为角度 θ 的精度。一般情况下，使用的精度是 π/180，表示要搜索所有可能的角度
    #  threshold 是阈值。该值越小，判定出的直线就越多
    # 返回值 lines 中的  每个元素  都是一对浮点数，表示检测到的直线的参数，即（r,θ），是 numpy.ndarray 类型
    lines = cv.HoughLines(edges, 1, np.pi/180, 200)
    # print(lines)
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = rho*a
        y0 = rho*b
        x1 = int(x0+1000*(-b))   # 这里的1000是指连续的1000像素内都有line上的点，就会识别为直线
        y1 = int(y0+1000*(a))
        x2 = int(x0-1000*(-b))
        y2 = int(y0-1000*(a))
        cv.line(image, (x1, y1), (x2, y2), (0, 255, 255), 1)
    image_dict = {"original": image_ori, "edges": edges, "霍夫直线检测": image}
    plt_pic(image_dict)


# 霍夫直线检测
# HoughLinesP是在HoughLines的基础上加了一个概率P，表明可以采用累积概率霍夫变换来找出二值图中的直线。
# HoughLinesP的第二个参数可以得到直线的两个端点坐标，这点是优势。累积概率霍夫变换相比于标准和多尺度计算量小，执行效率高
def line_detect_possible(path):
    image = cv.imread(path)

    if image is None:
        print("无法读取图片")
        return -1

    image_ori = image.copy()
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 50, 150, apertureSize=3)  # 上下限改成25,120，可以降低直线识别的门槛
    # minLineLength 用来控制「接受直线的最小长度」的值，默认值为 0。
    # maxLineGap 用来控制接受共线线段之间的最小间隔，即在一条线中两点的最大间隔
    # 返回值 lines 是由 numpy.ndarray 类型的元素构成的,返回值应该为：x1,y1,x2,y2
    lines = cv.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=50, maxLineGap=10)
    for line in lines:
        # print(type(line))
        x1, y1, x2, y2 = line[0]
        cv.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    image_dict = {"original": image_ori, "edges": edges, "霍夫直线检测S": image}
    plt_pic(image_dict)


# 霍夫圆检测
def detect_hough_circles(path):
    image = cv.imread(path)

    if image is None:
        print("无法读取图片")
        return -1

    image_ori = image.copy()
    dst = cv.pyrMeanShiftFiltering(image, 10, 100)    # 均值漂移算法是一种通用的聚类算法
    gray = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 20, 100, apertureSize=3)
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
    # dp=1最小步长，minDist是圆心之间的最小距离，minRadius默认为0，会自己去确认
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        cv.circle(image, (i[0], i[1]), i[2], (0, 0, 255), 2)
        cv.circle(image, (i[0], i[1]), 2, (255, 0, 0), 2)
    image_dict = {"original": image_ori, "edges": edges, "霍夫圆检测": image, "均值漂移": dst}
    plt_pic(image_dict)


if __name__ == "__main__":
    # image_path_lines = r"./images/hough_lines.jpg"
    # line_detection(image_path_lines)
    # line_detect_possible(image_path_lines)

    image_path_circle = r"./images/circle.png"
    detect_hough_circles(image_path_circle)
