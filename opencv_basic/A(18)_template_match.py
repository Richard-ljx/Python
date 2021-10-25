# 六种模板匹配算法(无法对图像旋转和缩放进行匹配)：平方差匹配法，相关匹配法，相关系数匹配法   及他们的归一化匹配法
# 模板匹配算法(？？？)：基于灰度值的模板匹配，基于形状的模板匹配，基于边缘特征点的模板匹配

import cv2 as cv
from image_merge import plt_pic


# 模板匹配算法
def template_demo(path1, path2):
    # target为待搜索图像，tpl为模板
    target = cv.imread(path1)
    tpl = cv.imread(path2)

    if (target is None) or (tpl is None):
        print("无法读取图片")
        return -1

    image_dict = {"模板图像": tpl, "待搜索图像": target}

    methods = [cv.TM_SQDIFF_NORMED, cv.TM_CCORR_NORMED, cv.TM_CCOEFF_NORMED]
    th, tw = tpl.shape[:2]
    for md in methods:
        if md == cv.TM_SQDIFF_NORMED:
            match = "归一化平方差匹配法"
        elif md == cv.TM_CCORR_NORMED:
            match = "归一化相关匹配法"
        elif md == cv.TM_CCOEFF_NORMED:
            match = "归一化相关系数匹配法"
        else:
            match = None
        result = cv.matchTemplate(target, tpl, md)    # 模板匹配结果，target为待搜索图像，tpl为模板
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)    # 查找对应的最小最大值和位置
        if md == cv.TM_SQDIFF_NORMED:
            tl = min_loc
        else:
            tl = max_loc
        br = (tl[0]+tw, tl[1]+th)
        cv.rectangle(target, tl, br, (0, 0, 255), thickness=2)
        image_dict[match] = target
    plt_pic(image_dict)


if __name__ == "__main__":
    image_path1 = r"./images/CrystalLiu2.jpg"
    image_path2 = r"./images/CrystalLiu22.jpg"
    template_demo(image_path1, image_path2)
