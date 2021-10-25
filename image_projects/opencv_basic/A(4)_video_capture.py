# 调用摄像头获取图片并显示

import cv2 as cv


def video_demo():
    capture = cv.VideoCapture(0)     # 0代表电脑自带摄像头

    if not capture.isOpened():
        print("摄像头打开失败")
        return -1

    while True:
        ret, frame = capture.read()  # ret代表帧读取是否正确的，正确为True，可用来判断一段视频是否读完
        # print(ret)
        frame = cv.flip(frame, 1)    # 调整摄像头方向，不然图像显示是镜像方向，“1”是左右调整，“0”是上下
        cv.imshow("video_test", frame)
        c = cv.waitKey(50)    # 每隔50ms显示一次图像
        if c == 27:           # 27==“Esc”
            break
    capture.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    video_demo()
