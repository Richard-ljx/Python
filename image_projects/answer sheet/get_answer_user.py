# 大致思路为：
# 对答题卡进行灰度->高斯滤波->Canny->透视变换->二值->利用掩膜进行与操作->查找每个掩膜提取结果非零值最多的作为答案

# 导入工具包
import numpy as np
import cv2 as cv
import os
import glob        # 文件名模式匹配，不用遍历整个目录判断每个文件是不是符合
from image_merge import plt_pic

# np.set_printoptions(threshold=np.inf)


# 透视变换的四个坐标点
def order_points(pts):
	# 一共4个坐标点
	rect = np.zeros((4, 2), dtype = "float32")

	# 按顺序找到对应坐标0123分别是 左上，右上，右下，左下
	# 计算左上，右下
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	# 计算右上和左下
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	return rect


# 透视变换
def four_point_transform(image, pts):
	# 获取输入坐标点
	rect = order_points(pts)
	(tl, tr, br, bl) = rect

	# 计算输入的w和h值
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	# 变换后对应坐标位置
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype="float32")

	# 计算变换矩阵
	M = cv.getPerspectiveTransform(rect, dst)
	warped = cv.warpPerspective(image, M, (maxWidth, maxHeight))

	# 返回变换后结果
	return warped


# 按坐标点排序（从左到右、从右到左、从上到下、从下到上）
def sort_contours(cnts, method="left-to-right"):
	reverse = False
	i = 0
	if method == "right-to-left" or method == "bottom-to-top":
		reverse = True
	if method == "top-to-bottom" or method == "bottom-to-top":
		i = 1
	boundingBoxes = [cv.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))

	return cnts, boundingBoxes


# 答题卡识别主程序
def get_answer(img_path, correct_answer):

	# 初始化需要显示的图像
	image_dict = {}

	# 预处理
	image = cv.imread(img_path)

	if image is None:
		print("无法读取图片")
		return -1

	image_dict['原始图'] = image
	contours_img = image.copy()

	gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
	blurred = cv.GaussianBlur(gray, (5, 5), 0)
	image_dict['高斯滤波'] = blurred
	edged = cv.Canny(blurred, 75, 200)
	image_dict['边缘检测'] = edged

	# 轮廓检测
	cnts = cv.findContours(edged, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
	cv.drawContours(contours_img, cnts, -1, (0, 0, 255), 3)
	image_dict['轮廓图'] = contours_img

	docCnt = None
	# 确保检测到了
	if len(cnts) > 0:
		# 根据轮廓大小进行排序
		cnts = sorted(cnts, key=cv.contourArea, reverse=True)

		# 遍历每一个轮廓
		for c in cnts:
			# 近似
			peri = cv.arcLength(c, True)
			approx = cv.approxPolyDP(c, 0.02 * peri, True)

			# 准备做透视变换
			if len(approx) == 4:
				docCnt = approx
				break

	# 执行透视变换
	warped = four_point_transform(gray, docCnt.reshape(4, 2))
	image_dict['透视变换'] = warped
	# 自适应阈值处理
	thresh = cv.threshold(warped, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
	image_dict['二值图'] = thresh
	thresh_Contours = thresh.copy()
	# 找到每一个圆圈轮廓
	cnts = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
	cv.drawContours(thresh_Contours, cnts, -1, 255, 3)
	image_dict['二值轮廓图'] = thresh_Contours

	questionCnts = []
	# 遍历
	for c in cnts:
		# 计算比例和大小
		(x, y, w, h) = cv.boundingRect(c)
		ar = w / float(h)

		# 根据实际情况指定标准
		if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
			questionCnts.append(c)

	# 按照从上到下进行排序
	questionCnts = sort_contours(questionCnts, method="top-to-bottom")[0]
	correct = 0

	# 每排有5个选项
	for (q, i) in enumerate(np.arange(0, len(questionCnts), 5)):
		# 排序
		cnts = sort_contours(questionCnts[i:i + 5])[0]
		bubbled = None

		# 遍历每一个结果
		for (j, c) in enumerate(cnts):
			# 使用mask来判断结果
			mask = np.zeros(thresh.shape, dtype="uint8")
			cv.drawContours(mask, [c], -1, 255, -1)       # 最后的-1表示填充
			# image_dict['掩膜'] = mask
			# 通过计算非零点数量来算是否选择这个答案
			mask_res = cv.bitwise_and(thresh, thresh, mask=mask)
			total = cv.countNonZero(mask_res)
			# image_dict['掩膜结果'] = mask_res
			# plt_pic(image_dict, arrangement=0)

			# 通过阈值判断
			if total < 350:    # 如果任何答案都未涂或者涂的不够多，则认为选题错误
				continue
			if bubbled is None or total > bubbled[0]:
				bubbled = (total, j)

		# 对比正确答案
		color = (0, 0, 255)
		k = correct_answer[q]

		# 判断正确
		if bubbled is None:
			correct = 0.0
		elif k == bubbled[1]:
			color = (0, 255, 0)
			correct += 1

		# 绘图
		cv.drawContours(warped, [cnts[k]], -1, color, 3)


	length = len(correct_answer)
	score = (correct / length) * 100
	print("[INFO] score: {:.2f}%".format(score))
	cv.putText(warped, "{:.2f}%".format(score), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
	image_dict['结果'] = warped

	plt_pic(image_dict, arrangement=0)


if __name__ == "__main__":
	# 正确答案:values中 0：A, 1：B, 2：C, 3：D, 4：E
	ANSWER_KEY = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}
	images = glob.glob(r'./images/*.png')
	for image in images:
		print(image)
		get_answer(image, ANSWER_KEY)
