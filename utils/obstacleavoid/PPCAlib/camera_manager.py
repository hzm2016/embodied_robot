"""
摄像头管理器模块
负责摄像头的初始化和管理（支持RGB和RGBD相机）
"""

import cv2
import numpy as np
import pyrealsense2 as rs
from typing import Tuple, Optional, Any


class CameraManager:
    """摄像头管理器 - 负责RGB和RGBD摄像头的初始化和管理"""
    
    def __init__(self, camera_id=0, width=640, height=480, 
                 color_resolution=(1280, 720), depth_resolution=(640, 480), fps=30):
        # RGB相机参数
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.cap = None
        
        # RGBD相机参数
        self.color_resolution = color_resolution
        self.depth_resolution = depth_resolution
        self.fps = fps
        self.pipeline = None
        self.config = None
        self.align = None
        self.use_rgbd = False
        
    def _find_l515(self, verbose: bool = True) -> bool:
        """检测L515相机"""
        devices = rs.context().query_devices()
        if len(devices) == 0:
            if verbose:
                print("未找到任何RealSense设备")
            return False

        found = False
        for dev in devices:
            name = dev.get_info(rs.camera_info.name)
            if verbose:
                print(f"找到设备：{name}")
            if "L515" in name.upper():
                found = True
        if verbose and found:
            print("✅ 成功识别L515相机！")
        if verbose and not found:
            print("⚠️ 已连接RealSense但未检测到L515型号")
        return found
        
    def initialize(self, use_rgbd=False):
        """初始化摄像头"""
        self.use_rgbd = use_rgbd
        
        if use_rgbd:
            return self._initialize_rgbd()
        else:
            return self._initialize_rgb()
    
    def _initialize_rgb(self):
        """初始化RGB摄像头"""
        self.cap = cv2.VideoCapture(self.camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        if not self.cap.isOpened():
            raise RuntimeError("Error: Could not open RGB camera.")
        
        print("RGB Camera opened successfully!")
        return True
    
    def _initialize_rgbd(self):
        """初始化RGBD摄像头"""
        if not self._find_l515(verbose=False):
            raise RuntimeError("未检测到L515，请检查连接后重试。")
        
        self.align = rs.align(rs.stream.color)
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 
                                 self.depth_resolution[0], self.depth_resolution[1], 
                                 rs.format.z16, self.fps)
        self.config.enable_stream(rs.stream.color, 
                                 self.color_resolution[0], self.color_resolution[1], 
                                 rs.format.bgr8, self.fps)
        
        try:
            self.pipeline.start(self.config)
            print("RGBD Camera (L515) opened successfully!")
            return True
        except Exception as e:
            raise RuntimeError(f"Error: Could not open RGBD camera: {e}")
    
    def read_frame(self):
        """读取一帧图像（RGB模式）"""
        if not self.use_rgbd:
            if self.cap is None:
                return False, None
            return self.cap.read()
        else:
            # RGBD模式下只返回彩色图像
            ret, frame, _, _ = self.read_rgbd_frame()
            return ret, frame
    
    def read_rgbd_frame(self):
        """读取RGBD一帧图像（包含深度信息）"""
        if not self.use_rgbd or self.pipeline is None:
            return False, None, None, None
        
        try:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                return False, None, None, None
            
            # 获取深度内参
            depth_intrinsics = depth_frame.profile.as_video_stream_profile().get_intrinsics()
            
            # 转换为numpy数组
            color_image = np.asanyarray(color_frame.get_data())
            
            return True, color_image, depth_frame, depth_intrinsics
            
        except Exception as e:
            print(f"Error reading RGBD frame: {e}")
            return False, None, None, None
    
    def robot_camera_coord(self):
        # 机械臂到相机相对位置
        # [-0.423, 1.439, 0.273]
        # [X, Z, Y]
        return np.array([0.092, 1.106, 0.338])
    
    def release(self):
        """释放摄像头"""
        if self.use_rgbd and self.pipeline is not None:
            self.pipeline.stop()
            print("RGBD Camera released.")
        elif not self.use_rgbd and self.cap is not None:
            self.cap.release()
            print("RGB Camera released.")