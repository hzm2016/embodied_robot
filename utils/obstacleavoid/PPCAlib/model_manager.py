"""
模型管理器模块
负责加载和管理所有AI模型（PoseFormer、HRNet等）
"""

import sys
import os
import argparse
import torch
import torch.nn as nn
from collections import OrderedDict

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sub_dir = os.path.join(project_root, 'PoseFormerV2')
sub_demo_dir = os.path.join(project_root, 'PoseFormerV2', 'demo')
sys.path.append(sub_dir)
sys.path.append(sub_demo_dir)

from PoseFormerV2.common.model_poseformer import PoseTransformerV2 as Model
from PoseFormerV2.demo.lib.hrnet.lib.config import cfg, update_config
from PoseFormerV2.demo.lib.hrnet.lib.models import pose_hrnet
from PoseFormerV2.demo.lib.yolov3.human_detector import load_model as yolo_model
from PoseFormerV2.demo.lib.sort.sort import Sort


class ModelManager:
    """模型管理器 - 负责加载和管理所有模型"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.poseformer_model = None
        self.poseformer_args = None
        self.hrnet_model = None
        self.human_model = None
        self.people_sort = None
        self.hrnet_args = None
        
    def load_poseformer_model(self):
        """加载PoseFormerV2模型"""
        args, _ = argparse.ArgumentParser().parse_known_args()
        args.embed_dim_ratio, args.depth, args.frames = 32, 4, 243
        args.number_of_kept_frames, args.number_of_kept_coeffs = 27, 27
        args.pad = (args.frames - 1) // 2
        args.previous_dir = 'checkpoint/'
        args.n_joints, args.out_joints = 17, 17

        model = nn.DataParallel(Model(args=args)).to(self.device)
        model_path = os.path.join(sub_dir, 'checkpoint', '27_243_45.2.bin')
        pre_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(pre_dict['model_pos'], strict=True)
        model.eval()
        
        self.poseformer_model = model
        self.poseformer_args = args
        
    def load_hrnet_model(self):
        """加载HRNet模型用于2D姿态估计"""
        cfg_dir = 'PoseFormerV2/demo/lib/hrnet/experiments/'
        model_dir = 'PoseFormerV2/demo/lib/checkpoint/'
        
        parser = argparse.ArgumentParser()
        parser.add_argument('--cfg', type=str, default=cfg_dir + 'w48_384x288_adam_lr1e-3.yaml')
        parser.add_argument('--modelDir', type=str, default=model_dir + 'pose_hrnet_w48_384x288.pth')
        parser.add_argument('--det-dim', type=int, default=416)
        parser.add_argument('--thred-score', type=float, default=0.50)
        parser.add_argument('--num-person', type=int, default=1)
        parser.add_argument('--gpu', type=str, default='0')
        parser.add_argument('opts', nargs=argparse.REMAINDER, default=None)
        args, _ = parser.parse_known_args()
        
        update_config(cfg, args)
        
        model = pose_hrnet.get_pose_net(cfg, is_train=False)
        model.to(self.device)
        state_dict = torch.load(args.modelDir, map_location=self.device)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
        model.eval()
        
        human_model = yolo_model(inp_dim=args.det_dim)
        people_sort = Sort(min_hits=0)
        
        self.hrnet_model = model
        self.human_model = human_model
        self.people_sort = people_sort
        self.hrnet_args = args
        
    def load_all_models(self):
        """加载所有模型"""
        print("Loading PoseFormerV2 model...")
        self.load_poseformer_model()
        print("PoseFormerV2 model loaded successfully!")
        
        print("Loading HRNet model...")
        self.load_hrnet_model()
        print("HRNet model loaded successfully!")