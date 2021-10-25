#导入工具包
from scipy.spatial import distance as dist
from collections import OrderedDict    # 这个模块实现了特定目标的容器，以提供Python标准内建容器 dict、list、set、tuple 的替代选择,详情见https://www.cnblogs.com/dianel/p/10787693.html#defaultdict
import numpy as np
import argparse
import time
import dlib
import cv2

# collections模块包含Counter、defaultdict、OrderedDict、namedtuple、deque、ChainMap等容器
# Counter：字典的子类，提供了可哈希对象的计数功能
# defaultdict：字典的子类，提供了一个工厂函数，为字典查询提供了默认值
# OrderedDict：字典的子类，保留了他们被添加的顺序, 但在python 3.7中，已经保证了python dict中元素的顺序和插入顺序是一致的
# namedtuple：创建命名元组子类的工厂函数
# deque：类似列表容器，实现了在两端快速添加(append)和弹出(pop)
# ChainMap：类似字典的容器类，将多个映射集合到一个视图里面
FACIAL_LANDMARKS_68_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 36)),
	("jaw", (0, 17))
])

# http://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf
def eye_aspect_ratio(eye):
	# 计算距离，竖直的
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	# 计算距离，水平的
	C = dist.euclidean(eye[0], eye[3])
	# ear值
	ear = (A + B) / (2.0 * C)
	return ear
 
# 输入参数
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="",
	help="path to input video file")
args = vars(ap.parse_args())
 
# 设置判断参数
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 3

# 初始化计数器
COUNTER = 0
TOTAL = 0

# 检测与定位工具
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# 分别取两个眼睛区域
(lStart, lEnd) = FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = FACIAL_LANDMARKS_68_IDXS["right_eye"]

# 读取视频
print("[INFO] starting video stream thread...")
vs = cv2.VideoCapture(args["video"])
#vs = FileVideoStream(args["video"]).start()
time.sleep(1.0)

def shape_to_np(shape, dtype="int"):
	# 创建68*2
	coords = np.zeros((shape.num_parts, 2), dtype=dtype)
	# 遍历每一个关键点
	# 得到坐标
	for i in range(0, shape.num_parts):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	return coords

# 遍历每一帧
while True:
	# 预处理
	frame = vs.read()[1]
	if frame is None:
		break
	
	(h, w) = frame.shape[:2]
	width=1200
	r = width / float(w)
	dim = (width, int(h * r))
	frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# 检测人脸
	rects = detector(gray, 0)

	# 遍历每一个检测到的人脸
	for rect in rects:
		# 获取坐标
		shape = predictor(gray, rect)
		shape = shape_to_np(shape)

		# 分别计算ear值
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)

		# 算一个平均的
		ear = (leftEAR + rightEAR) / 2.0

		# 绘制眼睛区域
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		# 检查是否满足阈值
		if ear < EYE_AR_THRESH:
			COUNTER += 1

		else:
			# 如果连续几帧都是闭眼的，总数算一次
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				TOTAL += 1

			# 重置
			COUNTER = 0

		# 显示
		cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(10) & 0xFF
 
	if key == 27:
		break

vs.release()
cv2.destroyAllWindows()