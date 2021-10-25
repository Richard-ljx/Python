# OpenCV+Tesserct-OCR  开操作 ==> 文字识别
# 验证码识别

import cv2 as cv
from PIL import Image
import pytesseract as tess
from image_merge import plt_pic


def OCR_yzm(img_path):
    image = cv.imread(img_path)

    if image is None:
        print("无法读取图片")
        return -1

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    kernel1 = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
    open1= cv.morphologyEx(binary, cv.MORPH_OPEN, kernel1)

    kernel2 = cv.getStructuringElement(cv.MORPH_RECT, (5, 1))
    open2 = cv.morphologyEx(open1, cv.MORPH_OPEN, kernel2)

    kernel3 = cv.getStructuringElement(cv.MORPH_RECT, (4, 1))
    open3 = cv.morphologyEx(open2, cv.MORPH_OPEN, kernel3)

    # cv2读出来的图片是BGR格式，matplotlib、Image读出来的图片是RGB格式
    textImage = Image.fromarray(open3)    # 将图像从ndarray类型转换成PIL.Image.Image类型
    text = tess.image_to_string(textImage)    # 将图片上的文字内容转化为文本字符串
    print("识别结果：%s" % text)

    image_dict = {"original": image, "二值图": binary, "2*2开操作": open1, "5*1开操作": open2, "4*1开操作": open3}
    plt_pic(image_dict)


if __name__ == "__main__":
    path = "./images/yzm.jpg"
    OCR_yzm(path)
