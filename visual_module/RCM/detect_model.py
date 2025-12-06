"""最简 YOLO 摄像头实时检测脚本
依赖: pip install ultralytics opencv-python
退出: 按 q 或 ESC
"""

from ultralytics import YOLO
import cv2
import numpy as np

class DetectModel:
	def __init__(self, model_path="best.pt", conf_threshold=0.5):
		self.model = YOLO(model_path)
		self.conf_threshold = conf_threshold

	def detect_result(self, frame):
		# 推理 (返回 list[Results])
		results = self.model(frame, verbose=False)  # 单张帧直接传 ndarray
		names = self.model.names
		return results, names
	
	def detect_and_visiualize(self, frame):
		results, names = self.detect_result(frame)
		r = results[0]
		boxes = r.boxes
		target = {"blue tag":"lesion1", "pink tag":"lesion2", "yellow tag":"lesion3", "camera":"tool", "tool1":"tool", "tool2":"tool"}
		
		overlay = frame.copy()
		alpha = 0.25          # box fill transparency
		for box in boxes:
			x1, y1, x2, y2 = box.xyxy[0].int().tolist()
			cls = int(box.cls[0])
			conf = float(box.conf[0])

			if conf < self.conf_threshold or y1+y2 < 300:  # 置信度阈值
				continue

			# ----- colors & style -----
			color = (0, 255, 0)  # BGR
			thickness = 2
			

			# ----- bbox with filled overlay -----
			cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)  # filled
			cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

			# ----- label text & background -----
			label_txt = f"{target[names.get(cls, cls)]} " if names else f"{cls} "
			(tw, th), baseline = cv2.getTextSize(
				label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
			)
			label_x1, label_y1 = x1, max(y1 - th - 8, 0)
			label_x2, label_y2 = x1 + tw + 8, y1

			# label background
			cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), color, -1)
			# label text (white)
			cv2.putText(
				frame,
				label_txt,
				(label_x1 + 4, label_y2 - 4),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.6,
				(0, 0, 0),
				2,
				cv2.LINE_AA,
			)

			# ----- target center circle for this box -----
			# cx = (x1 + x2) // 2
			# cy = (y1 + y2) // 2
			# cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1, cv2.LINE_AA)

		# blend overlay (semi‑transparent box fill)
		cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

		# ----- global crosshair in image center -----
		h, w = frame.shape[:2]
		cx, cy = w // 2, h // 2
		cross_len = 20
		cv2.line(frame, (cx - cross_len, cy), (cx + cross_len, cy), (0, 0, 255), 1, cv2.LINE_AA)
		cv2.line(frame, (cx, cy - cross_len), (cx, cy + cross_len), (0, 0, 255), 1, cv2.LINE_AA)
		cv2.circle(frame, (cx, cy), 10, (0, 0, 255), 1, cv2.LINE_AA)

		return frame
	
	def change_model(self, model_path):
		self.model = YOLO(model_path)

	def change_conf_threshold(self, new_threshold):
		self.conf_threshold = new_threshold

def main():
	model = DetectModel("best.pt")

	# 打开默认摄像头 0
	cap = cv2.VideoCapture(0)
	if not cap.isOpened():
		print("无法打开摄像头")
		return

	win_name = "Endo View"
	cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
	cavas = np.zeros((720, 1280, 3), dtype=np.uint8)

	while True:
		ret, frame = cap.read()
		if not ret:
			print("摄像头读取失败")
			break

		# 推理 (返回 list[Results])
		frame = model.detect_and_visiualize(frame)
		cv2.imshow(win_name, frame)
		key = cv2.waitKey(1) & 0xFF
		if key in (27, ord('q')):  # ESC 或 q 退出
			break

	cap.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()

