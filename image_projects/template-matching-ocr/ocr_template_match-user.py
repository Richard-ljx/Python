# 大致思路为：
# 1、将模板中的0~9数字提取出来作为后续的模板匹配中的模板使用(注意匹配的时候图片尺寸要一致)
# 2、对信用卡进行灰度->顶帽->Sobel(X方向，可不要)->闭操作->二值->闭操作->轮廓查找->条件筛选(数字宽高比)->匹配

# 信用卡识别
# from imutils import contours
import numpy as np
import cv2 as cv
import os
import glob        # 文件名模式匹配，不用遍历整个目录判断每个文件是不是符合
from matplotlib import pyplot as plt
from image_merge import plt_pic

# np.set_printoptions(threshold=np.inf)


# 指定信用卡类型
FIRST_NUMBER = {
	"3": "American Express",
	"4": "Visa",
	"5": "MasterCard",
	"6": "Discover Card"
}


def sort_contours(contours_list, method="left-to-right"):
	reverse = False   # 升序（默认）
	i = 0       # 比较x坐标

	if method == "right-to-left" or method == "bottom-to-top":
		reverse = True  # 降序
	if method == "top-to-bottom" or method == "bottom-to-top":
		i = 1   # 比较y坐标

	boundingBoxes = [cv.boundingRect(c) for c in contours_list]  # 用一个最小的矩形，把找到的形状包起来x,y,h,w
	# print(boundingBoxes[0])
	cont_bdg = zip(contours_list, boundingBoxes)
	# zip(*zip(a, b)代表解压成二维矩阵式, zip在python3.x后显示只能用list(zip(a,b))方法，且只能操作一次，内存即被释放
	(contours_list, boundingBoxes) = zip(*sorted(cont_bdg, key=lambda b: b[1][i], reverse=reverse))

	return contours_list, boundingBoxes


def resize(image, width=50, height=50, inter=cv.INTER_AREA):
	dim = None
	(h, w) = image.shape[:2]
	if width is None and height is None:
		return image
	if width is None:
		r = height / float(h)
		dim = (int(w * r), height)
	else:
		r = width / float(w)
		dim = (width, int(h * r))
	resized = cv.resize(image, dim, interpolation=inter)
	return resized


def num_template(img_path, width=57, height=88):
	# 读取一个模板图像
	img = cv.imread(img_path)
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	binary = cv.threshold(gray, 10, 250, cv.THRESH_BINARY_INV)[1]
	# contours1为查找到的轮廓，hierarchy为轮廓等级（Next, Previous, First Child, Parent），RETR_EXTERNAL为外轮廓
	contours1, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
	# print(contours1[0])26
	cv.drawContours(img, contours1, -1, (0, 0, 255), 3)
	boundingBoxes = sort_contours(contours1, method="left-to-right")[1]
	digits = {}
	image_dict = {"灰度图": img, "二值图": binary}
	# plt_pic(image_dict, arrangement=1)
	image_dict = {}
	for i, b in enumerate(boundingBoxes):
		x, y, _w, _h = b
		roi = binary[y:y+_h, x:x+_w]
		roi = cv.resize(roi, (width, height))
		digits[i] = roi
		image_dict[i] = roi
	# plt_pic(image_dict, arrangement=1)

	return digits


def card_recognize(img_path, width=57, height=88):
	# 提取0~9的数字，为模板匹配做准备
	base_path = os.path.dirname(img_path)

	template_path = os.path.join(base_path, "ocr_a_reference.png")
	digits = num_template(template_path, width, height)

	# 初始化需要显示的图像
	image_dict = {}

	# 读取一个卡片图像
	img = cv.imread(img_path)

	if img is None:
		print("无法读取图片")
		return -1

	img = resize(img, width=300)      # 保证每次识别的图片都是一个尺寸，方便后面的模板匹配
	image_dict["original"] = img
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	# 初始化卷积核
	rectKernel = cv.getStructuringElement(cv.MORPH_RECT, (9, 3))
	sqKernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))

	# open_op = cv.morphologyEx(gray, cv.MORPH_OPEN, rectKernel)
	# image_dict["开操作"] = open_op
	# 礼帽操作，突出更明亮的区域:当一幅图像具有大幅背景的时候，而微小物品比较有规律的情况下，可以使用顶帽操作进行背景提取。
	tophat = cv.morphologyEx(gray, cv.MORPH_TOPHAT, rectKernel)
	image_dict["顶帽"] = tophat

	# X梯度，实际中可不用这一步
	gradX = cv.Sobel(tophat, ddepth=cv.CV_32F, dx=1, dy=0, ksize=-1)     # ksize=-1相当于用3*3的
	# print(gradX.shape)
	gradX = np.absolute(gradX)
	minVal, maxVal = np.min(gradX), np.max(gradX)
	gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))        # 先归一化，再*255
	gradX = gradX.astype("uint8")
	image_dict["X梯度"] = gradX

	# 通过闭操作（先膨胀，再腐蚀）将数字连在一起
	close_op = cv.morphologyEx(gradX, cv.MORPH_CLOSE, rectKernel)
	image_dict["闭操作"] = close_op
	# THRESH_OTSU会自动寻找合适的阈值，适合双峰，需把阈值参数设置为0
	thresh = cv.threshold(close_op, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
	image_dict["二值图"] = thresh

	close2 = cv.morphologyEx(thresh, cv.MORPH_CLOSE, sqKernel)  # 再来一个闭操作
	# close2 = cv.morphologyEx(close2, cv.MORPH_CLOSE, sqKernel)  # 再来一个闭操作
	image_dict["第二次闭操作"] = close2

	threshCnts, hierarchy = cv.findContours(close2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)   # 计算轮廓

	cur_img = img.copy()

	locs = []
	# 遍历轮廓
	for i, c in enumerate(threshCnts):
		# 计算矩形
		(x, y, w, h) = cv.boundingRect(c)
		cv.rectangle(cur_img, (x, y), (x + w, y + h), (0, 255, 0), 1)
		ar = w / float(h)

		# 选择合适的区域，根据实际任务来，这里的基本都是四个数字一组
		if ar > 2.5 and ar < 4.0:

			if (w > 40 and w < 55) and (h > 10 and h < 20):
				# 符合的留下来
				locs.append((x, y, w, h))
	image_dict["轮廓图"] = cur_img

	# 将符合的轮廓从左到右排序
	locs = sorted(locs, key=lambda x: x[0])
	# print(locs)

	output = []
	res = img.copy()
	for (i, (gX, gY, gW, gH)) in enumerate(locs):
		# initialize the list of group digits
		groupOutput = []

		# 根据坐标提取每一个组
		group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
		# 预处理
		group = cv.threshold(group, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
		# 计算每一组的轮廓
		digitCnts, hierarchy = cv.findContours(group, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
		boundingBoxes = sort_contours(digitCnts,  method="left-to-right")[1]

		# 计算每一组中的每一个数值
		for b in boundingBoxes:
			# 找到当前数值的轮廓，resize成合适的的大小
			(x, y, w, h) = b
			roi = group[y:y + h, x:x + w]
			roi = cv.resize(roi, (width, height))

			# 计算匹配得分
			scores = []

			# 在模板中计算每一个得分
			for digitROI in digits.values():
				# 模板匹配: result: 返回匹配结果的相关程度
				result = cv.matchTemplate(roi, digitROI, cv.TM_CCOEFF)
				(_, score, _, _) = cv.minMaxLoc(result)  # 返回最小值、最大值及他们的索引位置
				scores.append(score)

			# 得到最合适的数字
			groupOutput.append(str(np.argmax(scores)))   # np.argmax用于取出最大位置的索引

		# 画出来
		cv.rectangle(res, (gX - 5, gY - 5),  (gX + gW + 5, gY + gH + 5), (0, 0, 255), 1)
		cv.putText(res, "".join(groupOutput), (gX, gY - 15), cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

		# 得到结果
		output.extend(groupOutput)    # extend()方法只接受一个列表作为参数，并将该参数的每个元素都添加到原有的列表中
	image_dict["识别结果"] = res

	# 打印结果
	print("Credit Card Type: {}".format(FIRST_NUMBER[output[0]]))
	print("Credit Card #: {}".format("".join(output)))

	plt_pic(image_dict, arrangement=0)


if __name__ == "__main__":
	images = glob.glob(r'./images/*.png')
	for image in images:
		card_recognize(image, 57, 88)

