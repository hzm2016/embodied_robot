"""
模型管理器模块
负责加载和管理YOLO姿态估计模型
"""

import os
import torch
from ultralytics import YOLO


class ModelManager:
    """模型管理器 - 负责加载和管理YOLO模型"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.yolo_model = None
        
    def load_yolo_model(self, model_path="../checkpoint/yolo11n-pose.pt"):
        """
        加载YOLO姿态估计模型
        
        Args:
            model_path: YOLO模型文件路径，默认为 "yolo11n-pose.pt"
        """
        if not os.path.exists(model_path):
            print(f"Warning: Model file {model_path} not found. Using default model.")
        
        try:
            self.yolo_model = YOLO(model_path)
            print(f"YOLO model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            raise
        
    def load_all_models(self):
        """加载所有模型"""
        print("Loading YOLO pose estimation model...")
        self.load_yolo_model()
        print("YOLO model loaded successfully!")