# 图像逻辑运算：加减乘除、求平均值和方差、与或非异或

import cv2 as cv
import numpy as np
from image_merge import plt_pic


# 图像加减乘除运算，矩阵对应数字各自相加减，即要求图像维度相同才可
def img_calculator(path1, path2):
    img1 = cv.imread(path1)
    img2 = cv.imread(path2)

    if (img1 is None) or (img2 is None):
        print("无法读取图片")
        return -1

    if img1.shape == img2.shape:
        dst_add = cv.add(img1, img2)              # 加运算
        dst_subtract = cv.subtract(img1, img2)    # 减运算
        dst_divide = cv.divide(img1, img2)        # 除运算
        dst_multiply = cv.multiply(img1, img2)    # 乘运算
        image_dict = {"original1": img1, "original2": img2, "dst_add": dst_add, "dst_subtract": dst_subtract,
                      "dst_divide": dst_divide, "dst_multiply": dst_multiply}
        plt_pic(image_dict)
    else:
        print("图像维度不相同")


# 图像求平均、方差运算
def others_mean_std(path):
    img2 = cv.imread(path)

    if img2 is None:
        print("无法读取图片")
        return -1

    m = cv.mean(img2)                   # 图像各个通道求平均运算
    print(m)
    std, dev = cv.meanStdDev(img2)      # 图像各个通道求平均值和方差运算
    print(std)
    print(dev)
    h, w = img2.shape[:2]
    img3 = np.zeros([h, w], np.uint8)
    std1, dev1 = cv.meanStdDev(img3)
    print(std1)
    print(dev1)


# 图像逻辑运算
def img_logic(path1, path2):
    img1 = cv.imread(path1)
    img2 = cv.imread(path2)

    if (img1 is None) or (img2 is None):
        print("无法读取图片")
        return -1

    dst_and = cv.bitwise_and(img1, img2)    # 与运算
    dst_or = cv.bitwise_or(img1, img2)      # 或运算
    dst_xor = cv.bitwise_xor(img1, img2)    # 异或运算
    dst_not = cv.bitwise_not(img2)          # 非运算
    image_dict = {"original1": img1, "original2": img2, "dst_and": dst_and, "dst_or": dst_or,
                  "dst_xor": dst_xor, "dst_not": dst_not}
    plt_pic(image_dict)


if __name__ == "__main__":
    image_path1 = r"./images/01.jpg"
    image_path2 = r"./images/02.jpg"
    # img_calculator(image_path1, image_path2)
    # others_mean_std(image_path1)
    img_logic(image_path1, image_path2)
