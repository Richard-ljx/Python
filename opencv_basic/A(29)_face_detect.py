# 人脸识别：HAAR 与 LBP 获取数据

import cv2 as cv


# 人脸识别：HAAR 与 LBP 获取数据，此例用的是HAAR算法
def face_detect_demo(haar_path):
    face_detector = cv.CascadeClassifier(r"./images/haarcascade_frontalface_alt_tree.xml")

    if face_detector is None:
        print("文件打开失败")
        return -1

    capture = cv.VideoCapture(0)

    if not capture.isOpened():
        print("摄像头打开失败")
        return -1

    while (True):
        ret, frame = capture.read()
        frame = cv.flip(frame, 1)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.02, 5)
        for x, y, w, h in faces:
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv.imshow("face detect", frame)
        c = cv.waitKey(10)
        if c == 27:
            break


if __name__ == "__main__":
    path = r"./images/haarcascade_frontalface_alt_tree.xml"
    face_detect_demo(path)
