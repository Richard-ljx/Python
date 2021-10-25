

import cv2 as cv


# opencv已经实现了的追踪算法
OPENCV_OBJECT_TRACKERS = {
	"csrt": cv.TrackerCSRT_create,
	"kcf": cv.TrackerKCF_create,
	"boosting": cv.TrackerBoosting_create,
	"mil": cv.TrackerMIL_create,
	"tld": cv.TrackerTLD_create,
	"medianflow": cv.TrackerMedianFlow_create,
	"mosse": cv.TrackerMOSSE_create
}


def multi_obj_track(video_path, tracker):
	# 实例化OpenCV's multi-object tracker
	trackers = cv.MultiTracker_create()
	print(video_path)
	vs = cv.VideoCapture(video_path)

	if not vs.isOpened():
		print("视频打开失败")
		return -1

	# 视频流
	while True:
		# 取当前帧
		frame = vs.read()[1]
		# 到头了就结束
		if frame is None:
			break

		# resize每一帧
		(h, w) = frame.shape[:2]
		width=600
		r = width / float(w)
		dim = (width, int(h * r))
		frame = cv.resize(frame, dim, interpolation=cv.INTER_AREA)

		# 追踪结果
		(success, boxes) = trackers.update(frame)

		# 绘制区域
		for box in boxes:
			(x, y, w, h) = [int(v) for v in box]
			cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

		# 显示
		cv.imshow("Frame", frame)
		key = cv.waitKey(100) & 0xFF

		if key == ord("s"):
			# 选择一个区域，按s
			box = cv.selectROI("Frame", frame, fromCenter=False,
				showCrosshair=True)

			# 创建一个新的追踪器
			tracker = OPENCV_OBJECT_TRACKERS[tracker]()
			trackers.add(tracker, frame, box)

		# 退出
		elif key == 27:
			break
	vs.release()
	cv.destroyAllWindows()


if __name__ == "__main__":
	path = r'./videos/los_angeles.mp4'
	track = "kcf"
	multi_obj_track(path, track)
