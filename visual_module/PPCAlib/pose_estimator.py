import numpy as np
import pyrealsense2 as rs


class PoseEstimator:
    """姿态估计器 - 使用YOLO提取2D关键点并基于RGBD还原3D姿态"""

    def __init__(self, model_manager):
        self.model_manager = model_manager
    
    def get_2d_keypoints(self, frame):
        """使用YOLO获取2D关键点，同时返回带骨骼渲染的图像"""
        annotated_frame = frame.copy()
        try:
            results = self.model_manager.yolo_model(frame, verbose=False)
            
            if len(results) == 0:
                keypoints, scores = self._get_default_keypoints(frame)
                return keypoints, scores, annotated_frame

            annotated_frame = results[0].plot()

            if results[0].keypoints is None or len(results[0].keypoints) == 0:
                keypoints, scores = self._get_default_keypoints(frame)
                return keypoints, scores, annotated_frame
            
            keypoints = results[0].keypoints.data[0]
            kpts = keypoints[:, :2].cpu().numpy()
            kpts_sc = keypoints[:, 2:3].cpu().numpy()
            
            return kpts, kpts_sc, annotated_frame
        except Exception as e:
            print(f"Error in get_2d_keypoints: {e}")
            keypoints, scores = self._get_default_keypoints(frame)
            return keypoints, scores, annotated_frame
    
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
    
    def estimate_3d_pose_from_rgbd(self, input_2D_no, input_2D_sc, depth_frame, depth_intrinsics, img_size):
        """使用RGBD数据估计3D姿态"""
        pose_3d_17 = np.zeros((17, 3))
        confidence = input_2D_sc.squeeze()
        
        for i in range(17):
            x, y = int(input_2D_no[i, 0]), int(input_2D_no[i, 1])
            x = max(0, min(x, img_size[1] - 1))
            y = max(0, min(y, img_size[0] - 1))
            
            depth = depth_frame.get_distance(x, y)
            
            if depth > 0 and depth < 10.0:
                point_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x, y], depth)
                pose_3d_17[i] = point_3d
            else:
                pose_3d_17[i] = self._estimate_invalid_depth_point(
                    input_2D_no, depth_frame, depth_intrinsics, i, img_size)
                confidence[i] *= 0.5
        
        pose_3d_17 = self._post_process_rgbd_pose(pose_3d_17)
        return pose_3d_17, confidence
    
    def _estimate_invalid_depth_point(self, input_2D_no, depth_frame, depth_intrinsics, joint_idx, img_size):
        """对于深度无效的关键点，尝试从邻近像素估计深度"""
        x, y = int(input_2D_no[joint_idx, 0]), int(input_2D_no[joint_idx, 1])
        
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
        
        related_joints = self._get_related_joints(joint_idx)
        for related_idx in related_joints:
            rx, ry = int(input_2D_no[related_idx, 0]), int(input_2D_no[related_idx, 1])
            rx = max(0, min(rx, img_size[1] - 1))
            ry = max(0, min(ry, img_size[0] - 1))
            
            related_depth = depth_frame.get_distance(rx, ry)
            if related_depth > 0 and related_depth < 10.0:
                point_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x, y], related_depth)
                return point_3d
        
        return [0.0, 0.0, 0.0]
    
    def _get_related_joints(self, joint_idx):
        """获取与指定关节相关的关节索引"""
        joint_connections = {
            0: [1, 2], 1: [0, 3], 2: [0, 4], 3: [1], 4: [2],
            5: [6, 7, 11], 6: [5, 8, 12], 7: [5, 9], 8: [6, 10],
            9: [7], 10: [8], 11: [5, 12, 13], 12: [6, 11, 14],
            13: [11, 15], 14: [12, 16], 15: [13], 16: [14],
        }
        return joint_connections.get(joint_idx, [])
    
    def _RGBD_2_base_coords(self):
        """RGBD坐标系转换到基础坐标系"""
        return np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]])
    
    def _post_process_rgbd_pose(self, pose_3d_17):
        """对RGBD得到的3D姿态进行后处理"""
        joint_relation = {
            0: [11, 12], 7: [5, 6, 11, 12], 8: [5, 6], 9: [5, 6, 3, 4],
            10: [0, 1, 2], 11: [5], 12: [7], 13: [9], 14: [6],
            15: [8], 16: [10], 1: [11], 2: [12], 3: [15], 4: [12],
            5: [14], 6: [16],
        }
        
        tran_pose_17 = np.zeros_like(pose_3d_17)
        Rotat = self._RGBD_2_base_coords()
        for i in range(17):
            if len(joint_relation[i]) == 1:
                tran_pose_17[i] = Rotat @ pose_3d_17[joint_relation[i][0]]
            else:
                tran_pose_17[i] = Rotat @ np.mean([pose_3d_17[j] for j in joint_relation[i]], axis=0)
        
        return tran_pose_17
