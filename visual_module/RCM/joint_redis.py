import threading

import cv2
import numpy as np
import redis
import time
import json
from typing import Optional
from math import pi
import numpy as np
import cv2

class JointAngleListener:
	"""Background subscriber that keeps the latest robot joint angles."""

	def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0,
				 channel: str = "actual_joint_angle") -> None:
		self._redis_client = redis.StrictRedis(host=host, port=port, db=db)
		self._channel = channel 
		self._publish = 'target_position' 
		self._lock = threading.Lock()  
		self._latest_angles: Optional[np.ndarray] = None
		self._stop_event = threading.Event()
		self._thread: Optional[threading.Thread] = None

	def start(self) -> None:
		if self._thread and self._thread.is_alive():
			return
		self._stop_event.clear()
		self._thread = threading.Thread(target=self._listen, daemon=True)
		self._thread.start()

	def stop(self) -> None:
		self._stop_event.set()
		if self._thread is not None:
			self._thread.join(timeout=1.0)

	def get_latest_angles(self) -> Optional[np.ndarray]:
		with self._lock:
			if self._latest_angles is None:
				return None
			return self._latest_angles.copy()

	def publish_joint_angles(self, angles: np.ndarray) -> None:
		# 将关节角度转换为 JSON
		message = {"target_joint_angle": angles.flatten().tolist()} 
		self._redis_client.publish(self._publish, json.dumps(message))
		print(f"Published djoint angles: {message}")

	def _listen(self) -> None:
		pubsub = self._redis_client.pubsub()
		pubsub.subscribe(self._channel)
		try:
			for message in pubsub.listen():
				if self._stop_event.is_set():
					break
				if message.get("type") != "message":
					continue
				payload = message.get("data")
				if isinstance(payload, bytes):
					payload = payload.decode()
				if not payload:
					continue
				try:
					decoded = json.loads(payload)
				except json.JSONDecodeError:
					continue
				joint_angles = decoded.get("joint_angle")
				if joint_angles is None:
					continue
				with self._lock:
					self._latest_angles = np.asarray(joint_angles, dtype=np.float64)
		finally:
			try:
				pubsub.close()
			except Exception:
				pass

