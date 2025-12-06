from RCM.detect_model import DetectModel
from RCM.scope_control import ScopeControl
import cv2
import os
import sys
from typing import Any, Dict
from Robotic_Arm.rm_robot_interface import *
import numpy as np
from math import pi
from RCM.Kinematics import GetEnd2JointVel
import time

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

	Realman = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
	handle = Realman.rm_create_robot_arm("192.168.2.18", 8080, 3)
	arm_data = Realman.rm_get_arm_current_trajectory()
	print("now_degrees:",arm_data["data"])


	theta = np.array([0, pi/4, 0, pi/2, 0, -pi/4, -pi/6])
	res = Realman.rm_movej(np.rad2deg(theta),10,0,0,1)
	print("movej result:",res)
	time.sleep(1)

	while True:
		ret, frame = cap.read()
		if not ret:
			print("摄像头读取失败")
			break

		# 推理 (返回 list[Results])
		results, names = model.detect_result(frame)
		# 控制逻辑
		vend = scope_controller.control_scope(results, names)
		vjoint = GetEnd2JointVel(vend, theta)
		theta = theta + vjoint.flatten()*0.01  # 关节速度，单位rad/s
		res = Realman.rm_movej(np.rad2deg(theta),10,0,0,1) # NOW NEVER USE CANFD
		print("movej result:",res)
	
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