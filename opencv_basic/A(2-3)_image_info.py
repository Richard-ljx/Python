# 对比opencv、matplotlib、PIL打开图片的类型和方式，以及图像输入输出、图像信息获取
# 1、opencv、matplotlib都是ndarray类型，PIL是PIL.Image.Image类型
# 2、opencv是BGR格式，matplotlib、PIL是RGB格式


import cv2 as cv     # 导入open-cv库
import os
from matplotlib import pyplot as plt
from PIL import Image


# BGR格式，ndarray类型
def cv_image(path):
    # 图像显示及保存方式
    image = cv.imread(path)    # 读入图片

    if image is None:
        print("无法读取图片")
        return -1

    # 显示图片信息
    print(type(image))     # ndarray类型
    print(image.shape)     # 三维矩阵(a, b, 3)
    print(image.size)      # a*b*3(高*宽*3)相乘的数值
    print(image.dtype)     # numpy中的数组元素类型，一般为“uint8”
    print(image)

    # 显示以及保存图片
    cv.namedWindow("test", cv.WINDOW_NORMAL)
    cv.imshow("test", image)
    cv.waitKey(0)   # 代表图片每隔0ms显示一次，即一直显示
    cv.destroyAllWindows()   # 退出所有相关的windows窗口
    dir_path = os.path.dirname(path)
    new_path = os.path.join(dir_path, "test_cv.jpg")
    cv.imwrite(new_path, image)


# RGB格式，ndarray类型
def matplotlib_image(path):
    # 图像显示及保存方式
    image = plt.imread(path)    # 读入图片

    # 显示图片信息
    print(type(image))     # ndarray类型
    print(image.shape)     # 三维矩阵(a, b, 3)
    print(image.size)      # a*b*3(高*宽*3)相乘的数值
    print(image.dtype)     # numpy中的数组元素类型，一般为“uint8”
    print(image)

    # 显示以及保存图片
    plt.imshow(image)
    dir_path = os.path.dirname(path)
    new_path = os.path.join(dir_path, "test_matplotlib.jpg")
    plt.savefig(new_path)   # 需要在plt.show之前，才能保存图片
    plt.show()


# RGB格式，PIL.Image.Image类型
def PIL_image(path):
    # 图像显示及保存方式
    image = Image.open(path)    # 读入图片
    # image = Image.fromarray(image)

    # 显示图片信息
    print(type(image))     # numpy类型
    print(image.size)      # a*b(高*宽)
    print(image)

    # 显示以及保存图片
    image.show()
    dir_path = os.path.dirname(path)
    new_path = os.path.join(dir_path, "test_PIL.jpg")
    image.save(new_path)


if __name__ == "__main__":
    image_path = r"./images/lena.jpg"
    cv_image(image_path)
    matplotlib_image(image_path)
    PIL_image(image_path)
