# 特征匹配:
# 图像傅里叶变换后低频在四周、高频在中心(经过fftshift后，低频在中间，高频在四周)


import cv2 as cv
import numpy as np
from image_merge import plt_pic


# sift特征检测:
# 1.尺度空间的极值检测;2.特征点定位;3.特征方向赋值;4.特征点描述.
def sift_feature(img_path):
    img = cv.imread(img_path)

    if img is None:
        print("无法读取图片")
        return -1

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    sift = cv.xfeatures2d.SIFT_create()
    kps = sift.detect(gray, None)

    # 关键点kp的方法有：
    # pt(x,y):关键点的点坐标；
    # size():该关键点邻域直径大小；
    # angle:角度，表示关键点的方向，值为[零,三百六十)，负值表示不使用。
    # response:响应强度
    # octacv:从哪一层金字塔得到的此关键点。
    # class_id:当要对图片进行分类时，用class_id对每个关键点进行区分，默认为-1
    kps_pt = np.float32([kp.pt for kp in kps])
    print(kps_pt[:5])

    img1 = img.copy()
    # DRAW_MATCHES_FLAGS_DEFAULT：默认，只绘制特征点的坐标点，显示在图像上就是一个个小圆点，每个小圆点的圆心坐标都是特征点的坐标。
    # DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG：函数不创建输出的图像，而是直接在输出图像变量空间绘制，要求本身输出图像变量就是一个初始化好了的，size与type都是已经初始化好的变量。
    # DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS ：单点的特征点不被绘制。
    # DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS：绘制特征点的时候绘制的是一个个带有方向的圆，这种方法同时显示图像的坐标，size和方向，是最能显示特征的一种绘制方式。
    img1 = cv.drawKeypoints(img, kps, img1, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)    # 第一个img是为了给kps找对应特征点

    # 计算特征，也可以直接使用sift.detectAndCompute(gray, None)
    kp, des = sift.compute(gray, kps)
    kp1, des1 = sift.detectAndCompute(gray, None)

    image_dict = {"original": img, "sift特征图像": img1}
    plt_pic(image_dict)


# harris角点检测:
# 1.计算差分图像;2.高斯平滑;3.计算局部极值;4.确认角点.
def harris_feature(img_path):
    img = cv.imread(img_path)

    if img is None:
        print("无法读取图片")
        return -1

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # cv2.cornerHarris参数解释：
    # img - 数据类型为 float32 的输入图像
    # blockSize - 角点检测中要考虑的领域大小
    # ksize - Sobel 求导中使用的窗口大小
    # k - Harris 角点检测方程中的自由参数,取值参数为 [0,04,0.06]
    dst = cv.cornerHarris(gray, 2, 3, 0.04)
    print('dst.shape:', dst.shape)

    img1 = img.copy()
    print(dst > 0.01 * dst.max())
    img1[dst > 0.01 * dst.max()] = [0, 0, 255]

    image_dict = {"original": img, "harris特征": dst, "harris特征图像": img1}
    plt_pic(image_dict)


# 特征匹配
def feature_match(img_path1, img_path2, k=2):
    if k < 0:
        print('k应该为正整数')
        return
    assert isinstance(k, int), 'k应该为正整数'

    img1 = cv.imread(img_path1, 0)
    img2 = cv.imread(img_path2, 0)

    if (img1 is None) or (img2 is None):
        print("无法读取图片")
        return -1


    image_dict = {"original1": img1, "original2": img2}
    plt_pic(image_dict)

    sift = cv.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    if k == 1:
        # 1对1匹配
        # crossCheck表示两个特征点要互相匹，例如A中的第i个特征点与B中的第j个特征点最近的，
        # 并且B中的第j个特征点到A中的第i个特征点也是NORM_L2: 归一化数组的(欧几里德距离)，
        # 如果其他特征计算方法需要考虑不同的匹配计算方式
        bf = cv.BFMatcher(crossCheck=True)  # 暴力匹配
        matches = bf.match(des1, des2)
        # match的方法有:
        # queryIdx：测试图像的特征点描述符的下标（第几个特征点描述符），同时也是描述符对应特征点的下标。
        # trainIdx：样本图像的特征点描述符下标,同时也是描述符对应特征点的下标。
        # distance：代表匹配的特征点描述符的欧式距离，数值越小也就说明俩个特征点越相近。
        matches = sorted(matches, key=lambda x: x.distance)
        img3 = cv.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)

    elif k > 0:
        # k对最佳匹配
        bf = cv.BFMatcher()  # 暴力匹配
        matches = bf.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])
        img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)

    else:
        return

    image_dict = {"sift特征匹配图像": img3}
    plt_pic(image_dict)


if __name__ == "__main__":
    path = r"./images/test_1.jpg"
    path1 = r"./images/box.png"
    path2 = r"./images/box_in_scene.png"
    # sift_feature(path)
    # harris_feature(path)
    feature_match(path1, path2, k=2)

