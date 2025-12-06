"""
关节映射器模块
将17关节姿态映射到10关节用于碰撞检测
"""

import numpy as np


class JointMapper:
    """关节映射器 - 将17关节映射到10关节用于碰撞检测"""
    
    def __init__(self):
        # PoseFormer 17关节定义：
        # 0: 骨盆, 1: 右髋, 2: 右膝, 3: 右踝, 4: 左髋, 5: 左膝, 6: 左踝
        # 7: 脊柱, 8: 颈部, 9: 头顶, 10: 左肩, 11: 左肘, 12: 左腕
        # 13: 右肩, 14: 右肘, 15: 右腕, 16: 颈底
        
        # 碰撞检测需要的10关节：
        # 0: 腰部中心, 1: 胸部, 2: 颈部, 3: 头部
        # 4: 左肩, 5: 左肘, 6: 左手
        # 7: 右肩, 8: 右肘, 9: 右手
        
        self.joint_mapping = {
            0: 7,   # 腰部中心 <- 骨盆
            1: 8,   # 胸部 <- 脊柱  
            2: 9,   # 颈部 <- 颈部
            3: 10,   # 头部 <- 头顶
            4: 11,  # 左肩 <- 左肩
            5: 12,  # 左肘 <- 左肘
            6: 13,  # 左手 <- 左腕
            7: 14,  # 右肩 <- 右肩
            8: 15,  # 右肘 <- 右肘
            9: 16   # 右手 <- 右腕
        }
    
    def map_joints(self, pose_17_joints):
        """
        将17关节姿态映射到10关节
        
        Args:
            pose_17_joints: 形状为(17, 3)的关节位置数组
            
        Returns:
            pose_10_joints: 形状为(10, 3)的关节位置数组
        """
        if pose_17_joints.shape[0] != 17:
            raise ValueError(f"输入应该有17个关节，但得到了{pose_17_joints.shape[0]}个")
        
        pose_10_joints = np.zeros((10, 3))
        
        for target_idx, source_idx in self.joint_mapping.items():
            pose_10_joints[target_idx] = pose_17_joints[source_idx]
        
        return pose_10_joints