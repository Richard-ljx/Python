# 分水岭算法，原理见：https://blog.csdn.net/fengye2two/article/details/79116105

import cv2 as cv
import numpy as np
from image_merge import plt_pic


# 分水岭算法
def watershed_demo(path):
    image = cv.imread(path)

    if image is None:
        print("无法读取图片")
        return -1

    image_ori = image.copy()
    print(image.shape)
    # 模糊->灰度->二值化
    blurred = cv.pyrMeanShiftFiltering(image, 10, 100)
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    mb = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv.dilate(mb, kernel, iterations=3)

    dist = cv.distanceTransform(mb, cv.DIST_L2, 3)
    dist_output = cv.normalize(dist, 0, 1.0, cv.NORM_MINMAX)

    ret, surface = cv.threshold(dist, dist.max() * 0.6, 255, cv.THRESH_BINARY)
    surface_fg = np.uint8(surface)

    unknown = cv.subtract(sure_bg, surface_fg)
    ret, markers = cv.connectedComponents(surface_fg)
    print("ret = %s" % ret)

    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv.watershed(image, markers=markers)

    image[markers == -1] = [0, 0, 255]
    image_dict = {"original": image_ori, "gray": gray, "binary": binary, "mor_opt": sure_bg,
                  "distance_t": dist_output * 50, "surface_bin": surface, "result": image}
    plt_pic(image_dict)


if __name__ == "__main__":
    image_path = r"./images/watershed.png"
    watershed_demo(image_path)
