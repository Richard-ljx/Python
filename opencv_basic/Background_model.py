# 背景建模:


import cv2 as cv


def background_model(video_path):
    #经典的测试视频
    cap = cv.VideoCapture(video_path)

    if not cap.isOpened():
        print("视频打开失败")
        return -1

    #形态学操作需要使用
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
    #创建混合高斯模型用于背景建模
    fgbg = cv.createBackgroundSubtractorMOG2()

    while(True):
        ret, frame = cap.read()
        fgmask = fgbg.apply(frame)
        #形态学开运算去噪点
        fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)
        #寻找视频中的轮廓
        contours, hierarchy = cv.findContours(fgmask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        for c in contours:
            #计算各轮廓的周长
            perimeter = cv.arcLength(c,True)
            if perimeter > 188:
                #找到一个直矩形（不会旋转）
                x,y,w,h = cv.boundingRect(c)
                #画出这个矩形
                cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        cv.imshow('frame',frame)
        cv.imshow('fgmask', fgmask)
        k = cv.waitKey(150) & 0xff
        if k == 27:
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    path = r'./images/test.avi'
    background_model(path)

