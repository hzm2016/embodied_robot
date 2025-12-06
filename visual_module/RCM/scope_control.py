"""最简 YOLO 摄像头实时检测脚本
依赖: pip install ultralytics opencv-python
退出: 按 q 或 ESC
"""

import numpy as np
import cv2

class ScopeControl:
	def __init__(self, track_goal="pink", threshold=0.5):
		self.track_goal = track_goal
		self.conf_threshold = threshold
		self.zone = 20
		self.uc = 360
		self.vc = 360
		self.kp = 0.02
		self.scale = 1.5
	def change_goal(self, new_goal):
		target = {"one":"blue tag", "two":"pink tag", "three":"yellow tag", "gripper":"tool"}
		self.track_goal = target[new_goal]
	
	def control_scope(self, results, names):
		xc, yc = self._results_process(results, names)
		# print(f"目标 {self.track_goal} 坐标: x={xc}, y={yc}")
		print("target:", self.track_goal)
		if xc and yc:
			# if abs(xc - self.uc) > self.zone or abs(yc - self.vc) > self.zone:
			if np.linalg.norm([xc - self.uc, yc - self.vc]) > self.zone and yc > 150:
				w_scope_y = self.kp * (yc - self.vc)
				w_scope_z = -self.kp * (xc - self.uc) / self.scale
			else:
				w_scope_y = 0
				w_scope_z = 0
		else:
			w_scope_y = 0
			w_scope_z = 0
		
		# 缺少由相机到末端坐标系转化
		# print("track control:", [w_scope_y, w_scope_z, 0, 0])
		# return np.array([0, w_scope_y, w_scope_z, 0, 0, 0])
		return np.array([0, w_scope_y, w_scope_z, 0, 0, 0])

	def _results_process(self, results, names):
		r = results[0]
		boxes = r.boxes
		xg = []
		yg = []
		for box in boxes:
			cls = int(box.cls[0])
			conf = float(box.conf[0])
			if (self.track_goal in names.get(cls, cls))and conf >= self.conf_threshold:
				x1, y1, x2, y2 = box.xyxy[0].int().tolist()
				xc = (x1 + x2) / 2
				yc = (y1 + y2) / 2
				xg.append(xc)
				yg.append(yc)
		return np.mean(xg), np.mean(yg)

	def change_threshold(self, new_threshold):
		self.conf_threshold = new_threshold
