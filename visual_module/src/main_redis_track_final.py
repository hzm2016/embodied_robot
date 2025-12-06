import threading

import cv2
import numpy as np
import redis
import time
import json
from math import pi
from RCM.Kinematics import GetEnd2JointVel
from RCM.detect_model import DetectModel
from RCM.scope_control import ScopeControl
from Robotic_Arm.rm_robot_interface import *


REDIS_CONFIG = {
	"host": "localhost",
	"port": 6379,
	"db": 0,
	"channel": "voice_commands",
	"channel_din": "endo",
}
find_flag = False
track_flag = False
gripper_flag = False
din = 0

def _voice_command_listener(scope_controller: ScopeControl, exit_event: threading.Event,
			exit_command_event: threading.Event) -> None:
	"""Listen for Redis pub/sub commands and update tracking goal."""
	pubsub = None
	global track_flag
	global find_flag
	global gripper_flag
	try:
		r = redis.Redis(host=REDIS_CONFIG["host"], port=REDIS_CONFIG["port"], db=REDIS_CONFIG["db"])
		pubsub = r.pubsub()
		pubsub.subscribe(REDIS_CONFIG["channel"])
		print("Listening for voice command messages...")
		for message in pubsub.listen():
			if exit_event.is_set():
				break
			if message.get("type") != "message":
				continue
			command = message["data"].decode().strip()
			command = command.replace("'", '"')  # 简单替换，确保 JSON 格式正确
			if not command:
				continue
			print(f"Received command: {command}")
			command = json.loads(command)
			
			if command["action"] == "exit":
				track_flag = False
				find_flag = False
				gripper_flag = False
			elif command["action"] == "find_target":
				find_flag = True
				print(f"Finding enabled: {find_flag}")
			elif command["action"] == "track_target":
				track_flag = True
				find_flag = False
				scope_controller.change_goal(command['target'])
				print(f"Tracking goal updated to: {command['target']}")
				gripper_flag = command['target'] == 'gripper'
	finally:
		try:
			if pubsub is not None:
				pubsub.close()  # type: ignore[has-type]
		except Exception:
			pass

def _din_get_listener(exit_event: threading.Event):
	global din
	r = redis.StrictRedis(host=REDIS_CONFIG["host"], port=REDIS_CONFIG["port"], db=REDIS_CONFIG["db"])
	# pubsub = r.pubsub()
	while True:
		payload = r.get(REDIS_CONFIG["channel_din"])
		data = payload.decode('utf-8').split(',')
		din = float(data[3])-0.1955
		# print(f"Initial din command: {din}")


def _move_to_current_joint(realman: RoboticArm, theta: np.ndarray) -> None:
	"""Command the arm to hold the current joint configuration."""
	current_deg = np.rad2deg(theta).tolist()
	res = realman.rm_movej_canfd(current_deg, False, 0, 1, 50)
	print(f"Move to current joint result: {res}")

def waiting_traj(index):
	wy = pi * np.sin(index/2000 * 2 * pi)
	wz = pi * np.cos(index/2000 * 2 * pi)
	wy = 0
	vend = np.array([0, wy, wz, 0, 0, 0])
	return vend

def main() -> None:
	global track_flag
	global find_flag
	global gripper_flag
	global din

	default_model_path = "./checkpoint/detect.pt"
	gripper_model_path = "./checkpoint/detect_tool.pt"
	current_model_path = default_model_path
	model = DetectModel(current_model_path)
	scope_controller = ScopeControl(track_goal="blue tag", threshold=0.3)

	cap = cv2.VideoCapture(0)
	if not cap.isOpened():
		print("无法打开摄像头")
		return
	win_name = "Endo view"
	# cv2.namedWindow(win_name, cv2.WINDOW_FULLSCREEN)
	cv2.namedWindow(win_name, cv2.WINDOW_GUI_NORMAL)
	cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
	Realman = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
	realman_handle = Realman.rm_create_robot_arm("192.168.2.18", 8080, 3)
	print("Robot connection handle:", realman_handle)
	arm_data = Realman.rm_get_arm_current_trajectory()
	print("now_degrees:", arm_data["data"])
	theta = np.deg2rad(arm_data["data"])
	time.sleep(1)

	exit_event = threading.Event()
	exit_command_event = threading.Event()

	listener = threading.Thread(
		target=_voice_command_listener,
		args=(scope_controller, exit_event, exit_command_event),
		daemon=True,
	)
	listener_din = threading.Thread(
		target=_din_get_listener,
		args=(exit_event,),
		daemon=True,
	)
	listener.start()
	listener_din.start()

	i_traj = 0

	cavas = np.ones((1080, 1920, 3), dtype=np.uint8)*255

	try:
		while not exit_event.is_set():
			ret, frame = cap.read()
			if not ret:
				print("摄像头读取失败")
				break
			# frame = cv2.imread("realtimeframe.jpg")
			results, names = model.detect_result(frame)
			print("find_flag:", find_flag)
			print("track_flag:", track_flag)
			print("gripper_flag:", gripper_flag)
			print("din:", din)

			if track_flag is True and find_flag is False:
				target_model_path = gripper_model_path if gripper_flag else default_model_path
				if target_model_path != current_model_path:
					model.change_model(target_model_path)
					current_model_path = target_model_path
				vend = scope_controller.control_scope(results, names)
				vjoint = GetEnd2JointVel(vend, theta, din)

			elif find_flag is True:
				vend = waiting_traj(i_traj)
				i_traj = (i_traj + 1) % 2000
				vjoint = GetEnd2JointVel(vend, theta, din)*0.2	
			else:
				vjoint = np.zeros((7, 1))
			theta = theta + vjoint.flatten()*0.001  # 关节速度，单位rad/s
			print("theta:", theta.flatten())
			res = Realman.rm_movej_canfd(np.rad2deg(theta), False, 0, 1, 50)
			print("movej result:", res)
			frame = model.detect_and_visiualize(frame)
			frame = cv2.resize(frame, (960, 960))
			cavas[60:1020, 280:1240, :] = frame
			cv2.imshow(win_name, cavas)
			cv2.imwrite("realtimeframe.jpg", frame)
			key = cv2.waitKey(1) & 0xFF
			if key in (27, ord("q")):
				exit_event.set()
				break
			elif key == ord("f"):
				find_flag = not find_flag
				print(f"Toggled finding to: {find_flag}")
			elif key == ord("t"):
				track_flag = not track_flag
				find_flag = False
				print(f"Toggled tracking to: {track_flag}")
			elif key == ord("g"):
				gripper_flag = not gripper_flag
				print(f"Toggled gripper tracking to: {gripper_flag}")
	finally:
		exit_event.set()
		cap.release()
		cv2.destroyAllWindows()
		if exit_command_event.is_set():
			_move_to_current_joint(Realman, theta)
		listener.join(timeout=1.0)


if __name__ == "__main__":
	main()