# 对象检测(有问题，需重看)

import cv2 as cv
import numpy as np
from image_merge import plt_pic


# 手写字母检测
def measure_object(path):
    image = cv.imread(path)

    if image is None:
        print("无法读取图片")
        return -1

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    dst = cv.cvtColor(binary, cv.COLOR_GRAY2BGR)
    print("threshold = %s" % ret)
    contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours):
        area = cv.contourArea(contour)
        print("area = %s" % area)
        x, y, w, h = cv.boundingRect(contour)
        rate = min(w, h) / max(w, h)
        print("rectangle rate = %s" % rate)
        mm = cv.moments(contour)    # 求图像矩
        print(type(mm))
        if mm['m00'] == 0:
            break
        else:
            cx = mm['m10'] / mm['m00']
            cy = mm['m01'] / mm['m00']

        cv.circle(dst, (np.int(cx), np.int(cy)), 3, (0, 255, 255), -1)    # 3表示圆的半径
        # cv.rectangle(dst,(x,y),(x+w,y+h),(0,0,255),2)

        # 多边形拟合，参数2(epsilon)是一个距离值，表示多边形的轮廓接近实际轮廓的程度，值越小，越精确；参数3表示是否闭合。
        approxCurve = cv.approxPolyDP(contour, 4, True)
        print(approxCurve.shape)
        if approxCurve.shape[0] > 6:
            cv.drawContours(dst, contours, i, (0, 255, 255), 2)
        if approxCurve.shape[0] == 4:
            cv.drawContours(dst, contours, i, (0, 0, 255), 2)
        if approxCurve.shape[0] == 3:
            cv.drawContours(dst, contours, i, (255, 255, 0), 2)

    image_dict = {"original": image, "binary_image": binary, "手写字母检测": dst}
    plt_pic(image_dict)


if __name__ == "__main__":
    image_path = r"./images/handwriting.jpg"
    measure_object(image_path)
