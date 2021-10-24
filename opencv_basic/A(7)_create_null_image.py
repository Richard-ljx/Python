# 利用numpy创建像素值全是0(或全是1)的图像

import cv2 as cv
import numpy as np


# 创建一个全是0(或全是1)的图像
def create_photo():
    img = np.zeros([400, 400, 3], np.uint8)
    # img = np.ones([400, 400, 3], np.uint8)
    img[:, :, 0] = np.ones([400, 400]) * 255   # 把第一列元素全改为255
    # print(img)
    cv.imshow("test1", img)


if __name__ == "__main__":
    cv.namedWindow("test1", cv.WINDOW_AUTOSIZE)
    create_photo()
    cv.waitKey(0)
    cv.destroyAllWindows()
