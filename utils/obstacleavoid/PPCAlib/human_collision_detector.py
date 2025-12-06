"""
人体碰撞检测器模块
为人体骨骼实时捆绑胶囊，并检测与固定正方体的碰撞
"""

import numpy as np
import fcl
import time
from scipy.spatial.transform import Rotation as R
from .Kinematics import GetFKinematics

class HumanCollisionDetector:
    """
    人体骨骼碰撞检测库
    为人体骨骼实时捆绑胶囊，并检测与固定正方体的碰撞
    """
    
    def __init__(self, capsule_radius=0.1, cube_size=0.5, base_position=None):
        """
        初始化碰撞检测器
        
        Args:
            capsule_radius: 胶囊半径
            cube_size: 正方体边长
            base_position: 正方体中心位置 [x, y, z]
        """
        self.capsule_radius = capsule_radius
        self.cube_size = cube_size
        
        # 设置正方体位置，默认在原点前方
        if base_position is None:
            self.base_position = np.array([0.0, 0.0, 1.0])
        else:
            self.base_position = np.array(base_position)
        
        # 人体骨骼连接定义（基于10个关节）
        self.bone_connections = [
            (0, 1), (1, 2), (2, 3),  # 躯干连接
            (1, 4), (4, 5), (5, 6),  # 左臂连接  
            (1, 7), (7, 8), (8, 9)   # 右臂连接
        ]
        
        # 创建固定的正方体碰撞对象
        self.cube_obj = self._create_cube()
        
        # 存储人体胶囊碰撞对象
        self.human_capsules = []
        self.robotic_capsules = []
        # 碰撞管理器
        self.human_manager = fcl.DynamicAABBTreeCollisionManager()
        self.robotic_manager = fcl.DynamicAABBTreeCollisionManager()
        # 存储碰撞检测结果
        self.collision_result = None
        self.distance_result = None
        self.last_update_time = None
        
    def _create_cube(self):
        """创建固定正方体碰撞对象"""
        cube_geom = fcl.Box(self.cube_size, self.cube_size, self.cube_size)
        cube_transform = fcl.Transform(np.eye(3), self.base_position)
        return fcl.CollisionObject(cube_geom, cube_transform)
    
    def get_rotation_center(self, vector1, vector2):
        """
        计算两个向量之间的旋转矩阵和中心点
        
        Args:
            vector1: 起点3D坐标
            vector2: 终点3D坐标
            
        Returns:
            rotation: 旋转矩阵
            center: 中心点坐标
        """
        ab_vector = vector2 - vector1
        norm_ab = np.linalg.norm(ab_vector)
        if norm_ab < 1e-6:  # 避免除零错误
            return R.from_matrix(np.eye(3)), vector1
        
        unit_ab = ab_vector / norm_ab
        z_axis = np.array([0, 0, 1])
        dot_product = np.dot(z_axis, unit_ab)
        
        if abs(float(dot_product)) > 0.999:
            rotation_obj = R.from_matrix(np.eye(3))  # 若近似平行则恒等旋转
        else:
            rotation_obj = R.align_vectors([unit_ab], [z_axis])[0]  # 不平行则旋转矩阵
        
        center_obj = (vector1 + vector2) / 2
        return rotation_obj, center_obj
    
    def create_robotic_capsules(self, theta):
        """
        为机械臂骨骼创建胶囊碰撞对象
        
        Args:
            theta: 机械臂关节角度列表，长度为7
            
        Returns:
            robotic_capsules: 胶囊碰撞对象列表
        """
        if theta is None or len(theta) != 7:
            return []
        
        # 获取机械臂关键点位置
        _, T = GetFKinematics(theta)
        joint_positions = [self.base_position]
        # 0,2,4,6
        T2base = np.eye(4)
        T2base[:3, 3] = self.base_position
        for i in range(7):
            T2base = T2base @ T[i]
            if i%2 == 0:
                pos = np.array([T2base[0, 3], T2base[1, 3], T2base[2, 3]])
                joint_positions.append(pos)
        joint_positions = np.array(joint_positions)
        self.robotic_joint = joint_positions
        # print(joint_positions)
        robotic_capsules = []
        index = [1,2]
        for start_idx in index:
            end_idx = start_idx + 1
            if start_idx < len(joint_positions) and end_idx < len(joint_positions):
                start_pos = joint_positions[start_idx]
                end_pos = joint_positions[end_idx]
                
                # 计算旋转和中心
                rotation, center = self.get_rotation_center(start_pos, end_pos)
                
                # 创建胶囊
                length = np.linalg.norm(end_pos - start_pos)
                capsule_geom = fcl.Capsule(self.capsule_radius, length)
                capsule_transform = fcl.Transform(rotation.as_matrix(), center)
                capsule_obj = fcl.CollisionObject(capsule_geom, capsule_transform)
                
                robotic_capsules.append(capsule_obj)
        return robotic_capsules

    def create_human_capsules(self, joint_positions):
        """
        为人体骨骼创建胶囊碰撞对象

        Args:
            joint_positions: 人体关节位置数组，形状为 (num_joints, 3)

        Returns:
            human_capsules: 胶囊碰撞对象列表
        """

        if joint_positions is None or len(joint_positions) < 2:
            return []
        
        human_capsules = []
        
        for start_idx, end_idx in self.bone_connections:
            if start_idx < len(joint_positions) and end_idx < len(joint_positions):
                start_pos = joint_positions[start_idx]
                end_pos = joint_positions[end_idx]
                
                # 计算旋转和中心
                rotation, center = self.get_rotation_center(start_pos, end_pos)
                
                # 创建胶囊
                length = np.linalg.norm(end_pos - start_pos)
                capsule_geom = fcl.Capsule(self.capsule_radius, length)
                capsule_transform = fcl.Transform(rotation.as_matrix(), center)
                capsule_obj = fcl.CollisionObject(capsule_geom, capsule_transform)
                
                human_capsules.append(capsule_obj)
        
        return human_capsules
    
    def update_human_pose(self, joint_positions):
        """
        更新人体姿态并创建新的胶囊碰撞对象
        
        Args:
            joint_positions: 人体关节位置数组，形状为 (num_joints, 3)
        """
        # 创建新的胶囊碰撞对象
        self.human_capsules = self.create_human_capsules(joint_positions)
        
        # 更新碰撞管理器
        self.human_manager.clear()
        if self.human_capsules:
            self.human_manager.registerObjects(self.human_capsules)
            self.human_manager.setup()
        
        self.last_update_time = time.time()

    def update_robotic_pose(self, theta):
        """
        更新机械臂姿态并创建新的胶囊碰撞对象
        
        Args:
            theta: 机械臂关节角度列表, 长度为7
        """
        # 创建新的胶囊碰撞对象
        self.robotic_capsules = self.create_robotic_capsules(theta)
        # 更新碰撞管理器
        self.robotic_manager.clear()
        if self.robotic_capsules:
            self.robotic_manager.registerObjects(self.robotic_capsules)
            self.robotic_manager.setup()

        self.last_update_time = time.time()

    def check_collision(self):
        """
        检测人体与正方体的碰撞
        
        Returns:
            collision_result: 碰撞检测结果字典
        """
        if not self.human_capsules:
            return {
                'is_collision': False,
                'min_distance': float('inf'),
                'nearest_points': None,
                'collision_points': None,
                'update_time': self.last_update_time
            }
        
        # 碰撞检测
        cdata = fcl.CollisionData()
        self.human_manager.collide(self.robotic_manager, cdata, fcl.defaultCollisionCallback)
        
        # 距离计算
        ddata = fcl.DistanceData()
        self.human_manager.distance(self.robotic_manager, ddata, fcl.defaultDistanceCallback)

        print(ddata.result.min_distance)


        # 存储结果
        self.collision_result = {
            'is_collision': cdata.result.is_collision,
            'collision_points': cdata.result.contacts,
            'update_time': self.last_update_time
        }
        
        self.distance_result = {
            'min_distance': ddata.result.min_distance,
            'nearest_points': ddata.result.nearest_points,
            'update_time': self.last_update_time
        }
        
        # 合并结果
        result = {
            'is_collision': self.collision_result['is_collision'],
            'min_distance': self.distance_result['min_distance'],
            'nearest_points': self.distance_result['nearest_points'],
            'collision_points': self.collision_result['collision_points'],
            'update_time': self.last_update_time
        }
        # print(result)
        return result
    
    def get_robotics_vertices(self):
        return self.robotic_joint
    
    def get_capsule_data(self):
        """获取胶囊体的几何数据用于可视化"""
        capsule_data = []
        
        for start_idx, end_idx in self.bone_connections:
            if hasattr(self, 'current_joints') and self.current_joints is not None:
                if start_idx < len(self.current_joints) and end_idx < len(self.current_joints):
                    start_pos = self.current_joints[start_idx]
                    end_pos = self.current_joints[end_idx]
                    
                    capsule_data.append({
                        'start': start_pos,
                        'end': end_pos,
                        'radius': self.capsule_radius
                    })
        
        return capsule_data
    
    def set_current_joints(self, joints):
        """设置当前关节位置用于可视化"""
        self.current_joints = joints
    '''
    def set_base_position(self, new_position):
        """设置立方体新位置"""
        self.base_position = np.array(new_position)
        # 重新创建立方体碰撞对象
        self.cube_obj = self._create_cube()
        # 更新碰撞管理器
        self.cube_manager.clear()
        self.cube_manager.registerObjects([self.cube_obj])
        self.cube_manager.setup()
    
    def set_cube_size(self, new_size):
        """设置立方体新大小"""
        self.cube_size = new_size
        # 重新创建立方体碰撞对象
        self.cube_obj = self._create_cube()
        # 更新碰撞管理器
        self.cube_manager.clear()
        self.cube_manager.registerObjects([self.cube_obj])
        self.cube_manager.setup()
    '''