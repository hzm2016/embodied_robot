"""
姿态估计器模块
负责2D和3D姿态估计处理（支持RGB和RGBD模式）
"""

import sys
import os
import copy
import numpy as np
import torch
import pyrealsense2 as rs

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sub_dir = os.path.join(current_dir, 'PoseFormerV2')
sys.path.append(sub_dir)
sub_demo_dir = os.path.join(current_dir, 'PoseFormerV2', 'demo')
sys.path.append(sub_demo_dir)

from PoseFormerV2.demo.lib.preprocess import h36m_coco_format, revise_kpts
from PoseFormerV2.common.camera import *
from PoseFormerV2.demo.lib.hrnet.lib.utils.utilitys import plot_keypoint, PreProcess, write, load_json
from PoseFormerV2.demo.lib.hrnet.lib.config import cfg, update_config
from PoseFormerV2.demo.lib.hrnet.lib.utils.transforms import *
from PoseFormerV2.demo.lib.hrnet.lib.utils.inference import get_final_preds
from PoseFormerV2.demo.lib.yolov3.human_detector import yolo_human_det as yolo_det


class PoseEstimator:
    """姿态估计器 - 负责2D和3D姿态估计"""
    
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.joints_left = [4, 5, 6, 11, 12, 13]
        self.joints_right = [1, 2, 3, 14, 15, 16]
        
    def get_2d_keypoints(self, frame):
        """获取2D关键点"""
        # 使用YOLO进行人体检测
        bboxs, scores = yolo_det(frame, self.model_manager.human_model, 
                                reso=self.model_manager.hrnet_args.det_dim, 
                                confidence=self.model_manager.hrnet_args.thred_score)
        
        # 检测失败时的处理
        if bboxs is None or not bboxs.any():
            if hasattr(self, 'bboxs_pre'):
                bboxs = self.bboxs_pre
                scores = self.scores_pre
            else:
                return self._get_default_keypoints(frame)
        else:
            self.bboxs_pre = copy.deepcopy(bboxs)
            self.scores_pre = copy.deepcopy(scores)
        
        # 使用SORT进行人体跟踪
        people_track = self.model_manager.people_sort.update(bboxs)
        
        if people_track.shape[0] >= 1:
            people_track_ = people_track[-1, :-1].reshape(1, 4)
        else:
            return self._get_default_keypoints(frame)
        
        track_bboxs = [[round(i, 2) for i in list(bbox)] for bbox in people_track_]
        
        # 使用HRNet进行2D姿态估计
        with torch.no_grad():
            inputs, origin_img, center, scale = PreProcess(frame, track_bboxs, cfg, 1)
            inputs = inputs[:, [2, 1, 0]]
            
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            output = self.model_manager.hrnet_model(inputs)
            
            preds, maxvals = get_final_preds(cfg, output.clone().cpu().numpy(), 
                                           np.asarray(center), np.asarray(scale))
        
        kpts = preds[0]
        kpts_sc = maxvals[0] if maxvals is not None else np.zeros((kpts.shape[0], 1))
        return kpts, kpts_sc
    
    def _get_default_keypoints(self, frame):
        """返回默认关键点位置"""
        h, w = frame.shape[:2]
        default_kpts = np.array([
            [w//2, h//4], [w//2, h//3], [w//2-20, h//3+30], [w//2-40, h//3+60],
            [w//2-50, h//3+90], [w//2+20, h//3+30], [w//2+40, h//3+60],
            [w//2+50, h//3+90], [w//2, h//2], [w//2-15, h//2+40],
            [w//2-20, h//2+80], [w//2-25, h//2+120], [w//2+15, h//2+40],
            [w//2+20, h//2+80], [w//2+25, h//2+120], [w//2-10, h//3+10],
            [w//2+10, h//3+10]
        ])
        return default_kpts, np.ones((17, 1)) * 0.5
    
    def estimate_3d_pose(self, input_2D_no, input_2D_sc, img_size):
        """估计3D姿态"""
        # 准备输入数据
        input_2D_padded = np.repeat(input_2D_no[np.newaxis, :, :], 
                                   self.model_manager.poseformer_args.frames, axis=0)
        input_2D_sc_padded = np.repeat(input_2D_sc[np.newaxis, :, :], 
                                      self.model_manager.poseformer_args.frames, axis=0)
        
        # 格式转换和标准化
        input_2D_padded = input_2D_padded[np.newaxis, :, :, :]
        input_2D_sc_padded = input_2D_sc_padded.squeeze(-1)[np.newaxis, :, :]
        
        input_2D_padded, input_2D_sc_padded, valid_frames = h36m_coco_format(
            input_2D_padded, input_2D_sc_padded)
        input_2D = normalize_screen_coordinates(input_2D_padded, w=img_size[1], h=img_size[0])
        input_2D = input_2D.squeeze(0)
        
        # 数据增强
        input_2D_aug = copy.deepcopy(input_2D)
        input_2D_aug[:, :, 0] *= -1
        input_2D_aug[:, self.joints_left + self.joints_right] = input_2D_aug[:, self.joints_right + self.joints_left]
        input_2D = np.concatenate((np.expand_dims(input_2D, axis=0), np.expand_dims(input_2D_aug, axis=0)), 0)
        input_2D = input_2D[np.newaxis, :, :, :, :]
        
        # 转换为tensor并进行3D姿态估计
        input_2D = torch.from_numpy(input_2D.astype('float32'))
        
        with torch.no_grad():
            output_3D_non_flip = self.model_manager.poseformer_model(input_2D[:, 0])
            output_3D_flip = self.model_manager.poseformer_model(input_2D[:, 1])
            
            # 翻转处理
            output_3D_flip[:, :, :, 0] *= -1
            output_3D_flip[:, :, self.joints_left + self.joints_right, :] = \
                output_3D_flip[:, :, self.joints_right + self.joints_left, :]
            
            # 平均结果
            output_3D = (output_3D_non_flip + output_3D_flip) / 2
            output_3D[:, :, 0, :] = 0  # 设置根节点为原点
            post_out = output_3D[0, 0].cpu().detach().numpy()
            
            # 转换到世界坐标系
            rot = [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]
            rot = np.array(rot, dtype='float32')
            post_out = camera_to_world(post_out, R=rot, t=0)
            post_out[:, 2] -= np.min(post_out[:, 2])
            
        return post_out, input_2D_sc_padded[:, -1, :].squeeze()
    
    def estimate_3d_pose_from_rgbd(self, input_2D_no, input_2D_sc, depth_frame, depth_intrinsics, img_size):
        """
        使用RGBD数据估计3D姿态
        通过2D关键点位置在深度图中获取深度信息，直接计算3D坐标
        
        Args:
            input_2D_no: 2D关键点坐标 (17, 2)
            input_2D_sc: 2D关键点置信度 (17, 1)
            depth_frame: RealSense深度帧
            depth_intrinsics: 深度相机内参
            img_size: 图像尺寸 (height, width)
            
        Returns:
            pose_3d_17: 3D关键点坐标 (17, 3)
            confidence: 置信度 (17,)
        """
        pose_3d_17 = np.zeros((17, 3))
        confidence = input_2D_sc.squeeze()
        
        for i in range(17):
            # 获取2D关键点像素坐标
            x, y = int(input_2D_no[i, 0]), int(input_2D_no[i, 1])
            
            # 确保坐标在图像范围内
            x = max(0, min(x, img_size[1] - 1))
            y = max(0, min(y, img_size[0] - 1))
            
            # 从深度图获取深度值
            depth = depth_frame.get_distance(x, y)
            
            if depth > 0 and depth < 10.0:  # 有效深度范围 (10米内)
                # 使用RealSense API将2D像素坐标转换为3D坐标
                point_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x, y], depth)
                pose_3d_17[i] = point_3d
            else:
                # 深度无效时，使用临近关键点的深度或设置为0
                pose_3d_17[i] = self._estimate_invalid_depth_point(
                    input_2D_no, depth_frame, depth_intrinsics, i, img_size)
                confidence[i] *= 0.5  # 降低置信度
        
        # 后处理：调整坐标系
        pose_3d_17 = self._post_process_rgbd_pose(pose_3d_17)
        
        return pose_3d_17, confidence
    
    def _estimate_invalid_depth_point(self, input_2D_no, depth_frame, depth_intrinsics, joint_idx, img_size):
        """
        对于深度无效的关键点，尝试从邻近像素或相关关键点估计深度
        """
        x, y = int(input_2D_no[joint_idx, 0]), int(input_2D_no[joint_idx, 1])
        
        # 在关键点周围搜索有效深度
        search_radius = 5
        for r in range(1, search_radius + 1):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if dx == 0 and dy == 0:
                        continue
                    
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < img_size[1] and 0 <= ny < img_size[0]:
                        depth = depth_frame.get_distance(nx, ny)
                        if depth > 0 and depth < 10.0:
                            point_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [nx, ny], depth)
                            return point_3d
        
        # 如果周围都没有有效深度，尝试使用相关关节的深度
        related_joints = self._get_related_joints(joint_idx)
        for related_idx in related_joints:
            rx, ry = int(input_2D_no[related_idx, 0]), int(input_2D_no[related_idx, 1])
            rx = max(0, min(rx, img_size[1] - 1))
            ry = max(0, min(ry, img_size[0] - 1))
            
            related_depth = depth_frame.get_distance(rx, ry)
            if related_depth > 0 and related_depth < 10.0:
                # 使用相关关节的深度，但保持当前关节的x, y位置
                point_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x, y], related_depth)
                return point_3d
        
        # 最后备选：返回零坐标
        return [0.0, 0.0, 0.0]
    
    def _get_related_joints(self, joint_idx):
        """
        获取与指定关节相关的关节索引（用于深度估计）
        基于人体关节的连接关系
        """
        # COCO 17关节的连接关系
        joint_connections = {
            0: [1, 2],        # 鼻子 -> 眼睛
            1: [0, 3],        # 左眼 -> 鼻子、左耳
            2: [0, 4],        # 右眼 -> 鼻子、右耳
            3: [1],           # 左耳 -> 左眼
            4: [2],           # 右耳 -> 右眼
            5: [6, 7, 11],    # 左肩 -> 右肩、左肘、左髋
            6: [5, 8, 12],    # 右肩 -> 左肩、右肘、右髋
            7: [5, 9],        # 左肘 -> 左肩、左腕
            8: [6, 10],       # 右肘 -> 右肩、右腕
            9: [7],           # 左腕 -> 左肘
            10: [8],          # 右腕 -> 右肘
            11: [5, 12, 13],  # 左髋 -> 左肩、右髋、左膝
            12: [6, 11, 14],  # 右髋 -> 右肩、左髋、右膝
            13: [11, 15],     # 左膝 -> 左髋、左踝
            14: [12, 16],     # 右膝 -> 右髋、右踝
            15: [13],         # 左踝 -> 左膝
            16: [14],         # 右踝 -> 右膝
        }
        
        return joint_connections.get(joint_idx, [])
    def _RGBD_2_base_coords(self):
        """
        对RGBD坐标系转换到基础坐标系
        """
        return np.array([[-1, 0, 0],
                         [0, 0, -1],
                         [0, -1, 0]])

    def _post_process_rgbd_pose(self, pose_3d_17):
        """
        对RGBD得到的3D姿态进行后处理
        """
        # 设置根节点（髋部中心）为原点
        # if np.any(pose_3d_17[11]) and np.any(pose_3d_17[12]):  # 左右髋都有效
        #     root_pos = (pose_3d_17[11] + pose_3d_17[12]) / 2
        # elif np.any(pose_3d_17[11]):  # 只有左髋有效
        #     root_pos = pose_3d_17[11]
        # elif np.any(pose_3d_17[12]):  # 只有右髋有效
        #     root_pos = pose_3d_17[12]
        # else:
        #     # 使用躯干中心作为根节点
        #     if np.any(pose_3d_17[5]) and np.any(pose_3d_17[6]):  # 左右肩
        #         root_pos = (pose_3d_17[5] + pose_3d_17[6]) / 2
        #     else:
        #         root_pos = np.array([0, 0, 0])
        
        # # 将所有关节相对于根节点
        # pose_3d_17 = pose_3d_17 - root_pos
        
        # # 确保Z轴最小值为0（人在地面上）
        # if np.any(pose_3d_17[:, 2]):
        #     pose_3d_17[:, 2] -= np.min(pose_3d_17[:, 2])

        joint_relation = {
            0: [11, 12],
            7: [5, 6, 11, 12],
            8: [5, 6],
            9: [5, 6, 3, 4],
            10: [0, 1, 2],
            11: [5],
            12: [7],
            13: [9],
            14: [6],
            15: [8],
            16: [10],
            1: [11],
            2: [12],
            3: [15],
            4: [12],
            5: [14],
            6: [16],
        }

        tran_pose_17 = np.zeros_like(pose_3d_17)
        Rotat = self._RGBD_2_base_coords()
        for i in range(17):
            if len(joint_relation[i]) == 1:
                tran_pose_17[i] = Rotat @ pose_3d_17[joint_relation[i][0]]
            else:
                tran_pose_17[i] = Rotat @ np.mean([pose_3d_17[j] for j in joint_relation[i]], axis=0)

        # 10/15 需要R旋转矩阵转一下
        return tran_pose_17