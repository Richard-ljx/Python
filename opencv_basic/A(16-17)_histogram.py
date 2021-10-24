# 图像直方图算法：图像整体直方图、BGR三通道直方图、整体增强对比度、局部增强对比度

import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
from image_merge import plt_pic


# 图像整体直方图
def histogram(path):
    lean = cv.imread(path)

    if lean is None:
        print("无法读取图片")
        return -1

    print(lean.ravel())
    print(type(lean.ravel()))
    plt.hist(lean.ravel(), 256, [0, 256])      # ravel、flatten、squeeze都是将多为数据降为一维，第一个256表示直方图的柱数
    plt.show()


# BGR三通道直方图
def image_hist(path):
    lean = cv.imread(path)

    if lean is None:
        print("无法读取图片")
        return -1

    color = ('blue', 'green', 'red')
    print(type(lean))
    print(type([lean]))
    for i, color in enumerate(color):
        hist = cv.calcHist([lean], [i], None, [256], [0, 256])      # None为掩膜
        plt.plot(hist, color=color)
        # plt.xlim() 显示的是x轴的作图范围，同时plt.ylim() 显示的是y轴的作图范围，而 plt.xticks() 表达的是x轴的刻度内容的范围
        plt.xlim([0, 256])
    plt.show()


# 整体增强对比度
def equalHist_demo(path):
    lean = cv.imread(path)

    if lean is None:
        print("无法读取图片")
        return -1

    gray = cv.cvtColor(lean, cv.COLOR_BGR2GRAY)
    plt.figure('直方图对比')
    plt.hist(gray.ravel(), 256)
    dst = cv.equalizeHist(gray)
    plt.hist(dst.ravel(), 256)
    plt.show()
    image_dict = {"original": lean, "整体增强对比度": dst}
    plt_pic(image_dict)


# 局部增强对比度/自适应均衡化图像
def clahe_demo(path):
    lean = cv.imread(path)

    if lean is None:
        print("无法读取图片")
        return -1

    gray = cv.cvtColor(lean, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))   # clipLimit 颜色对比度的阈值，titleGridSize 进行像素均衡化的网格大小，即在多少网格下进行直方图的均衡化操作
    dst = clahe.apply(gray)
    image_dict = {"original": lean, "局部增强对比度": dst}
    plt_pic(image_dict)


def create_rgb_hist(path):
    image = cv.imread(path)

    if image is None:
        print("无法读取图片")
        return -1

    h, w, c = image.shape
    rgbHist = np.zeros([16*16*16, 1], np.float32)
    bsize = 256/16
    for row in range(h):
        for col in range(w):
            b = image[row, col, 0]
            g = image[row, col, 1]
            r = image[row, col, 2]
            index = np.int(b/bsize)*16+np.int(g/bsize)*16+np.int(r/bsize)    # 用不同的16进制方法，错开bgr重叠的可能性，从而方便比较
            rgbHist[np.int(index), 0] = rgbHist[np.int(index), 0] + 1
    # print(rgbHist)
    return rgbHist


def hist_compare(image1, image2):
    hist1 = create_rgb_hist(image1)
    hist2 = create_rgb_hist(image2)

    if (hist1 is None) or (hist2 is None):
        print("无法读取图片")
        return -1

    # 比较它们的巴氏距离、相关性、卡方
    match1 = cv.compareHist(hist1, hist2, cv.HISTCMP_BHATTACHARYYA)
    match2 = cv.compareHist(hist1, hist2, cv.HISTCMP_CORREL)
    match3 = cv.compareHist(hist1, hist2, cv.HISTCMP_CHISQR)
    print("巴氏距离:%s，相关性:%s，卡方:%s"%(match1, match2, match3))


if __name__ == "__main__":
    image_path1 = r"./images/lena.jpg"
    image_path2 = r"./images/01.jpg"
    # histogram(image_path1)
    # image_hist(image_path1)
    # equalHist_demo(image_path1)
    # clahe_demo(image_path1)
    # create_rgb_hist(image_path1)
    hist_compare(image_path1, image_path2)
