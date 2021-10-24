# contours轮廓检测

import cv2 as cv
from image_merge import plt_pic


# 检测轮廓内部完全填充(无高斯模糊、无提取边缘)
def contour_demo(path):
    image = cv.imread(path)

    if image is None:
        print("无法读取图片")
        return -1

    image_ori = image.copy()
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for i, contour in enumerate(contours):
        cv.drawContours(image, contours, i, (0, 0, 255), -1)  # 这里的-1代表填充内部轮廓

    image_dict = {"original": image_ori, "二值图": binary, "轮廓检测1": image}
    plt_pic(image_dict)


# 检测轮廓内部完全填充(高斯模糊、无提取边缘)
def contour_demo_2(path):
    image = cv.imread(path)

    if image is None:
        print("无法读取图片")
        return -1

    image_ori = image.copy()

    dst = cv.GaussianBlur(image, (3, 3), 0)

    gray = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for i, contour in enumerate(contours):
        cv.drawContours(image, contours, i, (0, 0, 255), 2)

    image_dict = {"original": image_ori, "高斯模糊": dst, "二值图": binary, "轮廓检测2": image}
    plt_pic(image_dict)


# 检测轮廓内部完全填充(高斯模糊、提取边缘)
def contour_demo_3(path):
    image = cv.imread(path)

    if image is None:
        print("无法读取图片")
        return -1

    image_ori = image.copy()

    blurred = cv.GaussianBlur(image, (3, 3), 0)
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    xgrad = cv.Sobel(gray, cv.CV_16SC1, 1, 0)
    ygrad = cv.Sobel(gray, cv.CV_16SC1, 0, 1)
    edge_canny = cv.Canny(xgrad, ygrad, 50, 150)
    ret, binary = cv.threshold(edge_canny, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

    contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for i, contour in enumerate(contours):
        cv.drawContours(image, contours, i, (0, 0, 255), 2)  # 这里线的粗细thickness=2或-1有不同的效果

    image_dict = {"original": image_ori, "二值图": binary, "轮廓检测3": image}
    plt_pic(image_dict)


if __name__ == "__main__":
    image_path = r"./images/circle.png"
    # contour_demo(image_path)
    # contour_demo_2(image_path)
    contour_demo_3(image_path)
