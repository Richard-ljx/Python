#导入工具包

import numpy as np
import dlib
import cv2 as cv

# https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/
# http://dlib.net/files/

FACIAL_LANDMARKS_68_IDXS = {
	"mouth": (48, 68),
	"right_eyebrow": (17, 22),
	"left_eyebrow": (22, 27),
	"right_eye": (36, 42),
	"left_eye": (42, 48),
	"nose": (27, 36),
	"jaw": (0, 17)
}


FACIAL_LANDMARKS_5_IDXS = {
	"right_eye": (2, 3),
	"left_eye": (0, 1),
	"nose": (4,)
}

def shape_to_np(shape, dtype="int"):
	# 创建68*2
	coords = np.zeros((shape.num_parts, 2), dtype=dtype)
	# 遍历每一个关键点
	# 得到坐标
	for i in range(0, shape.num_parts):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	return coords

def visualize_facial_landmarks(image, shape, colors=None, alpha=0.75):
	# 创建两个copy
	# overlay and one for the final output image
	overlay = image.copy()
	output = image.copy()
	# 设置一些颜色区域
	if colors is None:
		colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23),
			(168, 100, 168), (158, 163, 32),
			(163, 38, 32), (180, 42, 220)]
	# 遍历每一个区域
	for (i, name) in enumerate(FACIAL_LANDMARKS_68_IDXS.keys()):
		# 得到每一个点的坐标
		(j, k) = FACIAL_LANDMARKS_68_IDXS[name]
		pts = shape[j:k]
		# 检查位置
		if name == "jaw":
			# 用线条连起来
			for l in range(1, len(pts)):
				ptA = tuple(pts[l - 1])
				ptB = tuple(pts[l])
				cv.line(overlay, ptA, ptB, colors[i], 2)
		# 计算凸包
		else:
			hull = cv.convexHull(pts)
			cv.drawContours(overlay, [hull], -1, colors[i], -1)
	# 叠加在原图上，可以指定比例
	cv.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
	return output


def detect_face_parts(img_path, shape_predictor):
	# 加载人脸检测与关键点定位
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(shape_predictor)

	# 读取输入数据，预处理
	image = cv.imread(img_path)

	if image is None:
		print("无法读取图片")
		return -1

	(h, w) = image.shape[:2]
	width=500
	r = width / float(w)
	dim = (width, int(h * r))
	image = cv.resize(image, dim, interpolation=cv.INTER_AREA)
	gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

	# 人脸检测
	rects = detector(gray, 1)

	# 遍历检测到的框
	for (i, rect) in enumerate(rects):
		# 对人脸框进行关键点定位
		# 转换成ndarray
		shape = predictor(gray, rect)
		shape = shape_to_np(shape)

		# 遍历每一个部分
		for (name, (i, j)) in FACIAL_LANDMARKS_68_IDXS.items():
			clone = image.copy()
			cv.putText(clone, name, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

			# 根据位置画点
			for (x, y) in shape[i:j]:
				cv.circle(clone, (x, y), 3, (0, 0, 255), -1)

			# 提取ROI区域
			(x, y, w, h) = cv.boundingRect(np.array([shape[i:j]]))

			roi = image[y:y + h, x:x + w]
			(h, w) = roi.shape[:2]
			width=250
			r = width / float(w)
			dim = (width, int(h * r))
			roi = cv.resize(roi, dim, interpolation=cv.INTER_AREA)

			# 显示每一部分
			cv.imshow("ROI", roi)
			cv.imshow("Image", clone)
			cv.waitKey(0)

		# 展示所有区域
		output = visualize_facial_landmarks(image, shape)
		cv.imshow("Image", output)
		cv.waitKey(0)


if __name__ == "__main__":
	path = r'./images/liudehua.jpg'
	shape_predictor = "shape_predictor_68_face_landmarks.dat"
	detect_face_parts(path, shape_predictor)
