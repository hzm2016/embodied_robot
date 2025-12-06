"""
碰撞可视化器模块
负责2D、3D姿态和碰撞检测的可视化
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib


class CollisionVisualizer:
    """碰撞可视化器 - 负责2D、3D姿态和碰撞检测的可视化"""
    
    def __init__(self):
        self.skeleton_2d = [
            [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
            [5, 11], [6, 12], [5, 6], [5, 7], [6, 8],
            [7, 9], [8, 10], [1, 2], [0, 1], [0, 2],
            [1, 3], [2, 4]
        ]
        
        # 设置matplotlib后端支持交互
        try:
            matplotlib.use('Qt5Agg')  # 首选Qt后端
        except ImportError:
            try:
                matplotlib.use('TkAgg')  # 备选Tk后端
            except ImportError:
                matplotlib.use('Agg')    # 如果都不可用，回退到Agg
                print("Warning: Interactive backend not available, 3D view won't be draggable")
        
        # 初始化3D显示
        plt.ion()
        self.fig = plt.figure(figsize=(8, 6))
        self.fig.suptitle('3D Collision Detection (Drag to rotate view)', fontsize=12)
        self.ax = self.fig.add_subplot(111, projection='3d')  # 碰撞检测可视化
        
        # 启用3D交互（如果后端支持）
        try:
            self.ax.mouse_init()
        except:
            pass
        
        # 设置初始视角
        self.ax.view_init(elev=20, azim=45)
        
        # 显示3D窗口
        plt.show(block=False)
        
    def draw_2d_pose(self, frame, keypoints, keypoints_sc, threshold=0.5):
        """在图像上绘制2D人体姿态"""
        frame_copy = frame.copy()
        
        # 绘制骨骼连接线
        for line in self.skeleton_2d:
            pt1 = (int(keypoints[line[0]][0]), int(keypoints[line[0]][1]))
            pt2 = (int(keypoints[line[1]][0]), int(keypoints[line[1]][1]))
            if keypoints_sc[line[0]] > threshold and keypoints_sc[line[1]] > threshold:
                cv2.line(frame_copy, pt1, pt2, (0, 255, 0), 2)
        
        # 绘制关键点
        for i, kp in enumerate(keypoints):
            if keypoints_sc[i] > threshold:
                cv2.circle(frame_copy, (int(kp[0]), int(kp[1])), 5, (0, 0, 255), -1)
        
        return frame_copy
        
    def draw_collision_detection(self, collision_detector, joints_10, collision_result):
        """绘制碰撞检测可视化"""
        self.ax.cla()
        # 设置标题，根据碰撞状态显示不同颜色
        if collision_result['is_collision']:
            self.ax.set_title('Collision Detected!', color='red', fontweight='bold')
        else:
            self.ax.set_title(f'Safe - Distance: {collision_result["min_distance"]:.3f}', color='green')
        
        # 绘制立方体
        # self._draw_cube(self.ax, collision_detector)
        self._draw_robotics(self.ax, collision_detector)
        # 绘制人体胶囊体
        self._draw_capsules(self.ax, collision_detector, collision_result['is_collision'])
        
        # 绘制人体关节点
        if joints_10 is not None:
            self.ax.scatter(joints_10[:, 0], joints_10[:, 1], joints_10[:, 2], 
                           c='blue', s=30, marker='o', alpha=0.7)
        
        # 绘制最近点连线（如果没有碰撞）
        if not collision_result['is_collision'] and collision_result['nearest_points'] is not None:
            nearest_points = collision_result['nearest_points']
            self.ax.plot([nearest_points[0][0], nearest_points[1][0]],
                         [nearest_points[0][1], nearest_points[1][1]],
                         [nearest_points[0][2], nearest_points[1][2]], 
                         'r--', linewidth=2, alpha=0.7)
            
            # 标注距离
            mid_point = (nearest_points[0] + nearest_points[1]) / 2
            self.ax.text(mid_point[0], mid_point[1], mid_point[2], 
                         f'{collision_result["min_distance"]:.3f}m', 
                         fontsize=8, color='red')
        # 设置坐标轴
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        
        # 设置坐标轴范围
        if joints_10 is not None:
            all_points = np.vstack([joints_10, collision_detector.get_robotics_vertices()])
            self._set_axis_limits(self.ax, all_points)
    
    def _draw_robotics(self, ax, collision_detector):
        """绘制机械臂"""
        vertices = collision_detector.get_robotics_vertices()
        
        # 绘制机械臂的连接线
        if vertices is not None and len(vertices) > 1:
            ax.plot(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                    'orange', linewidth=3, alpha=0.8)
            
            # 绘制关节点
            ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                       c='purple', s=50, marker='o', alpha=0.9)
        

    def _draw_cube(self, ax, collision_detector):
        """绘制立方体"""
        vertices = collision_detector.get_cube_vertices()
        
        # 定义立方体的12条边
        edges = [
            [0, 1], [1, 3], [3, 2], [2, 0],  # 底面
            [4, 5], [5, 7], [7, 6], [6, 4],  # 顶面
            [0, 4], [1, 5], [2, 6], [3, 7]   # 竖直边
        ]
        
        # 绘制边框
        for edge in edges:
            points = vertices[edge]
            ax.plot3D(*points.T, 'k-', alpha=0.6, linewidth=2)
        
        # 绘制半透明面
        faces = [
            [vertices[0], vertices[1], vertices[3], vertices[2]],  # 底面
            [vertices[4], vertices[5], vertices[7], vertices[6]],  # 顶面
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # 前面
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # 后面
            [vertices[0], vertices[2], vertices[6], vertices[4]],  # 左面
            [vertices[1], vertices[3], vertices[7], vertices[5]]   # 右面
        ]
        
        ax.add_collection3d(Poly3DCollection(faces, facecolors='cyan', alpha=0.2, edgecolors='black'))
    
    def _draw_capsules(self, ax, collision_detector, is_collision):
        """绘制胶囊体"""
        capsule_data = collision_detector.get_capsule_data()
        
        # 根据是否碰撞选择颜色
        color = 'red' if is_collision else 'green'
        
        for capsule in capsule_data:
            start = capsule['start']
            end = capsule['end']
            radius = capsule['radius']
            
            # 绘制胶囊体的中轴线
            ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
                   color=color, linewidth=max(2, radius*50), alpha=0.8)
            
            # 简化：只绘制端点球体（使用scatter代替surface）
            ax.scatter([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
                      c=color, s=radius*500, alpha=0.6, marker='o')
    
    def _set_axis_limits(self, ax, points):
        """设置坐标轴范围"""
        if len(points) == 0:
            return
        ''' 
        x_range = points[:, 0].max() - points[:, 0].min()
        y_range = points[:, 1].max() - points[:, 1].min()
        z_range = points[:, 2].max() - points[:, 2].min()
        max_range = max(x_range, y_range, z_range, 1.0)  # 最小范围为1
        
        mid_x = (points[:, 0].max() + points[:, 0].min()) / 2
        mid_y = (points[:, 1].max() + points[:, 1].min()) / 2
        mid_z = (points[:, 2].max() + points[:, 2].min()) / 2
        
        ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
        ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
        ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
        '''
        ax.set_xlim(-1.0,0.4)
        ax.set_ylim(-0.5,0.5)
        ax.set_zlim(-0.2,0.6)
    
    def cleanup(self):
        """清理资源"""
        plt.ioff()
        plt.close()