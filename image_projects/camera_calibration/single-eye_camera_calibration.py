# 单目相机校正：测量相机焦距和便宜主要的原理是张正友标定法，测量畸变参数是Brown算法
import cv2 as cv
import numpy as np
import glob
import os
from image_merge import plt_pic


# 相机校正
def camera_calibration(img_path, corner_size):     # corner_size类型为(w_num, h_num)
    w_num, h_num = corner_size       # w_num, h_num分别是棋盘格模板长边和短边规格（角点个数）

    # 找棋盘格角点，阈值
    # criteria：(type,max_iter,epsilon)这是迭代终止条件。满足此条件后，算法迭代将停止。
    # a. type终止条件的类型；b. max_it er-一个整数，指定最大迭代次数； c. epsilon-要求的精度。
    # type具有3个标志，如下所示：
    #     cv.TERM_CRITERIA_EPS-如果达到指定的精度epsilon，则停止算法迭代。
    #     cv.TERM_CRITERIA_MAX_ITER-在指定的迭代次数max_iter之后停止算法。
    #     cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER-当满足上述任何条件时，停止迭代。
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)    # criteria = (1 + 2, 30, 0.001)

    # 世界坐标系中的棋盘格点,例如(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)，去掉Z坐标,记为二维矩阵,认为在棋盘格这个平面上Z=0
    objp = np.zeros((w_num*h_num,3), np.float32)     # 构造0矩阵，88行3列，用于存放角点的世界坐标
    objp[:, :2] = np.mgrid[0:w_num,0:h_num].T.reshape(-1, 2)    # np.mgrid[返回多维结构，常见的如2D图形，3D图形。  .T是转置

    # 储存棋盘格角点的世界坐标和图像坐标对
    objpoints = [] # 在世界坐标系中的三维点
    imgpoints = [] # 在图像平面的二维点
    images = glob.glob(img_path)
    for fname in images:
        img = cv.imread(fname)

        if img is None:
            print("无法读取(%s)图片"  %fname)
            return -1

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # 粗略找到棋盘格角点 这里找到的是这张图片中角点的亚像素点位置，共11×8 = 88个点，gray必须是8位灰度或者彩色图，（w,h）为角点规模
        ret, corners = cv.findChessboardCorners(gray, (w_num, h_num))    # corners.shape=(w*h, 1, 2)
        # 如果找到足够点对，将其存储起来
        if ret:
            # 精确找到角点坐标
            corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # 将角点在图像上显示
            cv.drawChessboardCorners(img, (w_num, h_num), corners, ret)
            # cv.namedWindow('findCorners', cv.WINDOW_NORMAL)
            # cv.imshow('findCorners', img)
            # key = cv.waitKey(0) & 0xFF     # 0xFF是十进制的255
            key = ord("o")

            print("请确定此张图片角点位置是否正确(正确请按“o”，否则请按“Esc”)")
            # 如果按下键盘的"o"键，则认为此张图角点检测正确
            if key == ord("o"):    # ord()函数主要用来返回对应字符的ascii码
                # 将正确的objp点放入objpoints中
                objpoints.append(objp)
                imgpoints.append(corners)

    if not objpoints or not imgpoints:
        return -1
    cv.destroyAllWindows()

    # 相机标定函数
    # ret表示的是重投影误差；mtx是相机的内参矩阵；dist表述的相机畸变参数；
    # rvecs表示标定棋盘格世界坐标系到相机坐标系的旋转参数：rotation vectors，需要进行罗德里格斯转换；
    # tvecs表示translation vectors，主要是平移参数。
    retval, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print("重投影误差:%f" %retval)

    # 重投影误差
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i],imgpoints2, cv.NORM_L2) ** 2
        total_error += error
    print( "重投影误差: {}".format(np.sqrt(total_error/(len(objpoints)*len(imgpoints2)))))

    # 返回值：内参矩阵， 畸变参数
    return (mtx, dist)


# 优化校正后的系数并对单张图片校正
def undistorted_image(img_path, corner_size):
    val = camera_calibration(img_path, corner_size)
    if val == -1:
        print("未检测到任何一张角点图片")
        return -1
    mtx, dist = val

    dirname = os.path.dirname(img_path)
    img1_path = os.path.join(dirname, 'left1.jpg')
    img1 = cv.imread(img1_path)
    h,  w = img1.shape[:2]
    # 是调节视场大小，为1时视场大小不变，小于1时缩放视场; alpha=0，视场会放大，alpha=1，视场不变
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h)) # 自由比例参数

    # 对所有图片进行去畸变，有两种方法实现分别为： undistort()和remap()
    dst = cv.undistort(img1, mtx, dist, None, newcameramtx)
    # mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
    # dst = cv.remap(img1, mapx, mapy, cv.INTER_LINEAR)

    # 根据前面ROI区域裁剪图片,利用下面的undistorted_video函数观看更明显
    x, y, w, h = roi
    dst_crop = dst[y:y+h, x:x+w]

    image_dict = {'原始图像': img1, '校正图像': dst, '裁剪后校正图像': dst_crop}
    plt_pic(image_dict, arrangement=0)


# 优化校正后的系数并对视频进行校正
def undistorted_video(img_path, corner_size):
    val = camera_calibration(img_path, corner_size)
    if val == -1:
        print("未检测到任何一张角点图片")
        return -1
    mtx, dist = val

    capture = cv.VideoCapture(0)  # 创建一个 VideoCapture 对象

    if not capture.isOpened():
        print("摄像头打开失败")
        return -1

    while True:
        ret, img = capture.read()
        h, w = img.shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

        # 对所有图片进行去畸变，有两种方法实现分别为： undistort()和remap();remap()效率高
        # dst = cv.undistort(img, mtx, dist, None, newcameramtx)
        mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
        dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
        cv.imshow('dst', dst)

        # 根据前面ROI区域裁掉图像周围的黑边
        x, y, w, h = roi
        dst = dst[y:y + h, x:x + w]
        cv.imshow('dst_crop', dst)

        key = cv.waitKey(20) & 0xFF
        if key == 27:
            break
    capture.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    path = r'./images/left*.jpg'
    undistorted_image(path, (9, 6))
    # undistorted_video(path, (9, 6))
