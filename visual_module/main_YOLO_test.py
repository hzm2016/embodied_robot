from RCM.detect_model import DetectModel
from RCM.scope_control import ScopeControl
import cv2


def main():
	model = DetectModel("./checkpoint/detect.pt")
	scope_controller = ScopeControl(track_goal="blue tag", threshold=0.5)

	# 打开默认摄像头 0
	cap = cv2.VideoCapture(0)
	if not cap.isOpened():
		print("无法打开摄像头")
		return

	win_name = "YOLO Detect"
	cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

	while True:
		ret, frame = cap.read()
		if not ret:
			print("摄像头读取失败")
			break

		# 推理 (返回 list[Results])
		results, names = model.detect_result(frame)

		# 控制逻辑
		scope_controller.control_scope(results, names)

		# 可视化
		frame = model.detect_and_visiualize(frame)

		cv2.imshow(win_name, frame)
		key = cv2.waitKey(1) & 0xFF
		if key in (27, ord('q')):  # ESC 或 q 退出
			break

	cap.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()