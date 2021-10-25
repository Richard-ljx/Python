# 大致思路为：
# 灰度->高斯滤波->Canny->轮廓查找->找出纸张的矩形四个顶点->找出变换后的四个顶点及图像宽和高->透视变换

# 导入工具包
import numpy as np
import cv2 as cv
from image_merge import plt_pic


def order_points(pts):
	# 一共4个坐标点
	rect = np.zeros((4, 2), dtype="float32")

	# 按顺序找到对应坐标0123分别是 左上，右上，右下，左下
	# 计算左上，右下
	s = pts.sum(axis=1)         # axis=1是行，axis=0是列
	rect[0] = pts[np.argmin(s)]   # 给出axis方向最小值的下标,默认展开成一维数组
	rect[2] = pts[np.argmax(s)]

	# 计算右上和左下
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	return rect


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


# INTER_NEAREST:最邻近插值; INTER_LINEAR:双线性插值（默认设置）;
# INTER_CUBIC:4x4像素邻域的双三次插值; INTER_LANCZOS4:8x8像素邻域的Lanczos插值;
# INTER_AREA:使用像素区域关系进行重采样，它可能是图像抽取的首选方法，因为它会产生无云纹理的结果。但是当图像缩放时，它类似于INTER_NEAREST方法。
# 将图像转换为规定尺寸的图像
def resize(image, width=None, height=None, inter=cv.INTER_AREA):
	dim = None
	(h, w) = image.shape[:2]
	if not width and not height:
		return image
	if not width:
		r = height / float(h)
		dim = (int(w * r), height)
	else:
		r = width / float(w)
		dim = (width, int(h * r))
	resized = cv.resize(image, dim, interpolation=inter)

	return resized


def scan_main(img_path, width=None, height=None):
	image_dict = {}
	img = cv.imread(img_path)

	if img is None:
		print("无法读取图片")
		return -1

	# 坐标也会相同变化
	ratio = img.shape[0] / height     # 不能放后面，因为后面img尺寸被改变了
	orig = img.copy()
	img = resize(orig, height=height)

	# 预处理
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	gray = cv.GaussianBlur(gray, (5, 5), 0, 0)     # 后两位0分别代表高斯函数在X和Y方向的偏差
	edged = cv.Canny(gray, 75, 200)

	# 展示预处理结果
	print("STEP 1: 边缘检测")
	image_dict["STEP 1: Image"] = img
	image_dict["STEP 1: Edged"] = edged

	# 轮廓检测,cnts为三维的，RETR_LIST检测所有的轮廓，不需要内外包含关系
	cnts = cv.findContours(edged.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)[0]
	cnts = sorted(cnts, key=cv.contourArea, reverse=True)[:5]    # 按照轮廓面积降序排序

	# 遍历轮廓
	for c in cnts:
		# 计算轮廓近似
		peri = cv.arcLength(c, True)     # True表示封闭的
		# 进行多边形逼近，得到多边形的角点，第2个参数epsilon表示从原始轮廓到近似轮廓的最大距离，它是一个准确度参数
		approx = cv.approxPolyDP(c, 0.02 * peri, True)

		# 4个点的时候就拿出来
		if len(approx) == 4:
			# print(approx)
			break
		else:
			print("未检测到轮廓")
			return

	# 展示结果
	print("STEP 2: 获取轮廓")
	img_draw = img.copy()
	cv.drawContours(img_draw, [approx], -1, (0, 255, 0), 2)
	image_dict["STEP 2: Outline"] = img_draw

	# 透视变换
	warped = four_point_transform(orig, approx.reshape(4, 2) * ratio)

	# 二值处理
	warped = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
	ref = cv.threshold(warped, 100, 255, cv.THRESH_BINARY)[1]
	# cv.imwrite('./scan.jpg', ref)
	# 展示结果
	print("STEP 3: 变换")
	image_dict["STEP 3: warped"] = resize(warped, height=height)    # 透视变换的图像
	image_dict["STEP 3: Scanned"] = resize(ref, height=height)      # 二值化后的图像
	plt_pic(image_dict, arrangement=1)


if __name__ == "__main__":
	path = "./images/page.jpg"
	scan_main(path, height=500)
