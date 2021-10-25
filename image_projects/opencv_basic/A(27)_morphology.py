# 膨胀：3*3的结构元素，max值去替换该结构元素中心值                    顶帽和黑帽有问题
# 腐蚀：用Min去替换中心像素

# 开操作=腐蚀+膨胀，去掉小噪点
# 闭操作=膨胀+腐蚀，去掉中间内部的小黑点，填充小的封闭区域

# 顶帽：原图像与开操作之间的差距
# 黑帽：闭操作与原图像之间的差距

import cv2 as cv
import numpy as np
from image_merge import plt_pic


# 腐蚀
def erode_demo(path):
    image = cv.imread(path)

    if image is None:
        print("无法读取图片")
        return -1

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    dst = cv.erode(binary, kernel)
    cv.imshow("erode", dst)
    print(kernel)
    image_dict = {"original": image, "binary": binary, "腐蚀": dst}
    plt_pic(image_dict)


# 膨胀
def dilate_demo(path):
    image = cv.imread(path)

    if image is None:
        print("无法读取图片")
        return -1

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    dst = cv.dilate(binary, kernel)
    print(kernel)
    image_dict = {"original": image, "binary": binary, "膨胀": dst}
    plt_pic(image_dict)


# 开操作
def open_demo(path):
    image = cv.imread(path)

    if image is None:
        print("无法读取图片")
        return -1

    print(image.shape)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 10))
    open_demo1 = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)
    image_dict = {"original": image, "binary": binary, "开操作": open_demo1}
    plt_pic(image_dict)



# 闭操作
def close_demo(path):
    image = cv.imread(path)

    if image is None:
        print("无法读取图片")
        return -1

    print(image.shape)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (8, 8))
    close_demo1 = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)
    image_dict = {"original": image, "binary": binary, "开操作": close_demo1}
    plt_pic(image_dict)


# 顶帽
def top_hat_demo(path):
    image = cv.imread(path)

    if image is None:
        print("无法读取图片")
        return -1

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    dst = cv.morphologyEx(gray, cv.MORPH_TOPHAT, kernel)
    cimage = np.array(gray.shape, np.uint8)    # 为了明显一些
    cimage = 100
    dst = cv.add(dst, cimage)
    image_dict = {"original": image, "gray": gray, "顶帽": dst}
    plt_pic(image_dict)


# 黑帽
def black_hat_demo(path):
    image = cv.imread(path)

    if image is None:
        print("无法读取图片")
        return -1

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    dst = cv.morphologyEx(gray, cv.MORPH_BLACKHAT, kernel)
    cimage = np.array(gray.shape, np.uint8)  # 为了明显一些
    cimage = 100
    dst = cv.add(dst, cimage)
    image_dict = {"original": image, "gray": gray, "黑帽": dst}
    plt_pic(image_dict)


if __name__ == "__main__":
    image_path = r"./images/circle.png"
    # erode_demo(image_path)
    # dilate_demo(image_path)
    # open_demo(image_path)
    # close_demo(image_path)
    top_hat_demo(image_path)
    # black_hat_demo(image_path)
