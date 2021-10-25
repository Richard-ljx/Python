# sift特征匹配（利用knn暴力匹配）
# 首先利用sift进行特征匹配，然后利用knn进行特征值暴力匹配，得到匹配的结果，然后将两张图片进行融合

import cv2 as cv
import numpy as np


class Stitcher:

    # 拼接函数
    def stitch(self, images, ratio=0.75, reprojThresh=4.0, showMatches=False):
        # 获取输入图片
        (imageB, imageA) = images
        # 检测A、B图片的SIFT关键特征点，并计算特征描述子
        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        (kpsB, featuresB) = self.detectAndDescribe(imageB)

        # 匹配两张图片的所有特征点，返回匹配结果
        M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)

        # 如果返回结果为空，没有匹配成功的特征点，退出算法
        if M is None:
            return None

        # 否则，提取匹配结果
        # H是3x3视角变换矩阵
        (matches, H, status) = M
        # 将图片A进行视角变换，result是变换后图片
        result = cv.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
        self.cv_show('result', result)
        # 将图片B传入result图片最左端
        result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
        self.cv_show('result', result)
        # 检测是否需要显示图片匹配
        if showMatches:
            # 生成匹配图片
            vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches, status)
            # 返回结果
            return (result, vis)

        # 返回匹配结果
        return result

    # 显示图片
    def cv_show(self, name, img):
        cv.imshow(name, img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    # 计算特征坐标及特征向量
    def detectAndDescribe(self, image):
        # 将彩色图片转换成灰度图
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # 建立SIFT生成器
        descriptor = cv.xfeatures2d.SIFT_create()
        # 检测SIFT特征点，并计算描述子,features为计算的特征
        (kps, features) = descriptor.detectAndCompute(gray, None)

        # 将结果转换成NumPy数组,pt(x,y):关键点的点坐标
        kps = np.float32([kp.pt for kp in kps])

        # 返回特征点集，及对应的描述特征
        return (kps, features)

    # 利用暴力匹配器对特征进行匹配
    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
        # 建立暴力匹配器
        matcher = cv.BFMatcher()

        # 使用KNN检测来自A、B图的SIFT特征匹配对，K=2
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)

        matches = []
        for m in rawMatches:
            # 当最近距离跟次近距离的比值小于ratio值时，保留此匹配对
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                # 存储两个点在featuresA, featuresB中的索引值
                matches.append((m[0].trainIdx, m[0].queryIdx))

        # 当筛选后的匹配对大于4时，计算视角变换矩阵
        if len(matches) > 4:
            # 获取匹配对的点坐标
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])

            # 计算视角变换矩阵, cv2.findHomography计算单应矩阵
            (H, status) = cv.findHomography(ptsA, ptsB, cv.RANSAC, reprojThresh)

            # 返回结果
            return (matches, H, status)

        # 如果匹配对小于4时，返回None
        return None

    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        # 初始化可视化图片，将A、B图左右连接到一起
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB

        # 联合遍历，画出匹配对
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # 当点对匹配成功时，画到可视化图上
            if s == 1:
                # 画出匹配对
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv.line(vis, ptA, ptB, (0, 255, 0), 1)

        # 返回可视化结果
        return vis


def imageSitch(img_path1, img_path2):
    # 读取拼接图片
    imageA = cv.imread(img_path1)
    imageB = cv.imread(img_path2)

    if (imageA is None) or (imageB is None):
        print("无法读取图片")
        return -1

    # 把图片拼接成全景图
    stitcher = Stitcher()
    (result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)

    # 显示所有图片
    cv.imshow("Image A", imageA)
    cv.imshow("Image B", imageB)
    cv.imshow("Keypoint Matches", vis)
    cv.imshow("Result", result)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    # 读取拼接图片
    path1 = "left_01.png"
    path2 = "right_01.png"
    imageSitch(path1, path2)