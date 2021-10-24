''' 边缘检测算法：Roberts、Prewitt、Sobel、Scharr、Laplacian、Canny  '''
'''Roberts:常用来处理具有陡峭的低噪声图像，当图像边缘接近于正45度或负45度时，该算法处理效果更理想。
           其缺点是对边缘的定位不太准确，提取的边缘线条较粗。dx=[[-1, 0], [0, 1]];dy=[[0, -1], [1, 0]]'''
'''Prewitt:边缘检测结果在水平方向和垂直方向均比Robert算子更加明显。适合用来识别噪声较多、灰度渐变的图像。
           dx=[[-1, -1, -1], [0, 0, 0], [1, 1, 1]];dy=[[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]'''
'''Sobel:结合了高斯平滑和微分求导(分化),结果会具有更多的抗噪性,当对精度要求不是很高时,Sobel算子是一种较为常用的边缘检测方法。
           dx=[[-1, -2, -1], [0, 0, 0], [1, 2, 1]];dy=[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]'''
'''Scharr:相比Sobel算子，有更高的精度，且运算速度并没有降低。 ddepth 的值应该设为cv2.CV_64F
          dx=[[-3, -10, -3], [0, 0, 0], [3, 10, 3]];dy=[[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]]'''
'''Laplacian:一种二阶导数算子，其具有旋转不变性，可以满足不同方向上的图像边缘锐化(边缘检测)的要求。
             常用于图像增强领域(通过从图像中减去它的Lapacian图像，可以增强图像的对比度)。
             其对噪声比较敏感，由于其算法可能会出现  双像素边界  ，常用来判断边缘像素位于图像的明区或暗区，很少用于边缘检测；
             [[0, 1, 0], [1, -4, 1], [0, 1, 0]]'''
'''Canny:相对于前面计中，比较好的边缘提取方法，最常用。
         共分为四步：1、高斯模糊(Canny算子中不包含)；2、利用Sobel求梯度；
                    3、非极大值抑制NMS(比梯度方向梯度最大值，细化边缘)；
                    4、双阈值筛选边缘(一般高阈值比低阈值为3:1或2:1比较好)'''

import cv2 as cv
import numpy as np
from image_merge import plt_pic


# Sobel算子
def sobel_grad_demo(path):
    image = cv.imread(path)

    if image is None:
        print("无法读取图片")
        return -1

    grad_x = cv.Sobel(image, cv.CV_32F, 1, 0)
    grad_y = cv.Sobel(image, cv.CV_32F, 0, 1)
    gradx = cv.convertScaleAbs(grad_x)    # 对参数取绝对值
    grady = cv.convertScaleAbs(grad_y)
    gradxy = cv.addWeighted(gradx, 0.5, grady, 0.5, 0)      # src = gradx^0.5 + grady^0.5 + 0
    image_dict = {"original": image, "grad_x": gradx, "grad_y": grady, "Sobel算子": gradxy}
    plt_pic(image_dict)


# Scharr算子
def scharr_grad_demo(path):
    image = cv.imread(path)

    if image is None:
        print("无法读取图片")
        return -1

    grad_x = cv.Scharr(image, cv.CV_32F, 1, 0)
    grad_y = cv.Scharr(image, cv.CV_32F, 0, 1)
    gradx = cv.convertScaleAbs(grad_x)
    grady = cv.convertScaleAbs(grad_y)
    gradxy = cv.addWeighted(gradx, 0.5, grady, 0.5, 0)
    image_dict = {"original": image, "grad_x": gradx, "grad_y": grady, "grad_xy": gradxy}
    plt_pic(image_dict)


# Laplacian算子
def laplacian_grad_demo(path):
    image = cv.imread(path)

    if image is None:
        print("无法读取图片")
        return -1

    dst = cv.Laplacian(image, cv.CV_32F)
    lpls = cv.convertScaleAbs(dst)
    lpls_reinforced = cv.subtract(image, lpls)
    image_dict = {"original": image, "自定义Laplacian算子": lpls, "Laplacian增强算子": lpls_reinforced}
    plt_pic(image_dict)


# 自定义Laplacian算子
def custom_laplacian_grad_demo(path):
    image = cv.imread(path)

    if image is None:
        print("无法读取图片")
        return -1

    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])   # [[1,1,1],[1,-8,1],[1,1,1]]增强的算子
    kernel_reinforced = np.array([[0, 1, 0], [1, -5, 1], [0, 1, 0]])
    dst = cv.filter2D(image, cv.CV_32F, kernel=kernel)
    dst_reinforced = cv.filter2D(image, cv.CV_32F, kernel=kernel_reinforced)
    lpls = cv.convertScaleAbs(dst)
    lpls_reinforced = cv.convertScaleAbs(dst_reinforced)
    image_dict = {"original": image, "自定义Laplacian算子": lpls, "Laplacian增强算子": lpls_reinforced}
    plt_pic(image_dict)


# Canny算子：先高斯滤波，再使用Sobel算子，再用Canny算子
def edge_demo(path):
    image = cv.imread(path)

    if image is None:
        print("无法读取图片")
        return -1

    blurred = cv.GaussianBlur(image, (3, 3), 0)
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)

    # xgrad、ygrad分别提取边缘
    xgrad = cv.Sobel(gray, cv.CV_16SC1, 1, 0)
    ygrad = cv.Sobel(gray, cv.CV_16SC1, 0, 1)
    edge_canny_xy = cv.Canny(xgrad, ygrad, 50, 150)
    dst = cv.bitwise_and(image, image, mask=edge_canny_xy)

    # xgrad+ygrad可以换成gray直接去canny
    edge_canny_all = cv.Canny(gray, 50, 150)
    dst1 = cv.bitwise_and(image, image, mask=edge_canny_all)

    image_dict = {"original": image, "edge_canny_demo_xy": edge_canny_xy, "Color Edge xy": dst,
                  "edge_canny_demo_all": edge_canny_all, "Color Edge all": dst1}
    plt_pic(image_dict)


if __name__ == "__main__":
    image_path = r"./images/lena.jpg"
    # sobel_grad_demo(image_path)
    # scharr_grad_demo(image_path)
    # laplacian_grad_demo(image_path)
    # custom_laplacian_grad_demo(image_path)
    edge_demo(image_path)

