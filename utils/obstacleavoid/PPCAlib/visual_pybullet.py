import pybullet as p
import pybullet_data
import time
import numpy as np


class RobotArmVisualizer:
    """简单的机械臂PyBullet可视化控制类"""
    # RealManmodel/rm_75_b_description.urdf
    # Roboticmodel/Assembly_v6_fixed.urdf
    def __init__(self, urdf_path="RealManmodel/rm_75_b_description.urdf"):
        """初始化PyBullet环境和机械臂
        
        Args:
            urdf_path: URDF文件路径
        """
        # 连接PyBullet并启动GUI
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # 设置重力
        p.setGravity(0, 0, 0)
        
        # 加载地面
        p.loadURDF("plane.urdf")
        
        # 加载机械臂
        # 正常机械臂
        self.robot_id = p.loadURDF(urdf_path, [0, 0, 0], p.getQuaternionFromEuler([0, 0, np.pi/2]),useFixedBase=True)
        # 深圳机械臂
        # self.robot_id = p.loadURDF(urdf_path, [0, 0, 0], p.getQuaternionFromEuler([0, 0, np.pi]),useFixedBase=True)  
        # 获取关节信息
        self.num_joints = p.getNumJoints(self.robot_id)
        self.joint_indices = []
        
        self.bone_connections = [
            (0, 1), (1, 2), (2, 3),  # 躯干连接
            (1, 4), (4, 5), (5, 6),  # 左臂连接  
            (1, 7), (7, 8), (8, 9)   # 右臂连接
        ]

        # 找到可控制的关节
        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_type = joint_info[2]
            if joint_type == p.JOINT_REVOLUTE:
                self.joint_indices.append(i)
        
        print(f"找到 {len(self.joint_indices)} 个可控制关节")
        print(f"关节索引: {self.joint_indices}")
    
    def set_joint_angles(self, theta_values):
        """设置关节角度
        
        Args:
            theta_values: 关节角度列表 (弧度)
        """
        if len(theta_values) != len(self.joint_indices):
            print(f"警告: 输入角度数量 ({len(theta_values)}) 与关节数量 ({len(self.joint_indices)}) 不匹配")
            return
        
        # 设置关节目标位置
        for i, angle in enumerate(theta_values):
            p.setJointMotorControl2(
                self.robot_id,
                self.joint_indices[i],
                p.POSITION_CONTROL,
                targetPosition=angle
            )
    
    def get_joint_angles(self):
        """获取当前关节角度"""
        angles = []
        for joint_idx in self.joint_indices:
            joint_state = p.getJointState(self.robot_id, joint_idx)
            angles.append(joint_state[0])
        return angles
    
    def _create_bone_cylinder(self, start_pos, end_pos, radius=0.02, color=[0, 1, 0, 0.8]):
        """创建单个骨骼圆柱体"""
        bone_vector = end_pos - start_pos
        bone_length = np.linalg.norm(bone_vector)
        
        if bone_length < 0.01:
            return None
        
        # 圆柱体中心位置
        center_pos = (start_pos + end_pos) / 2
        
        # 计算圆柱体方向
        bone_unit = bone_vector / bone_length
        z_axis = np.array([0, 0, 1])
        
        # 计算旋转四元数
        if np.allclose(bone_unit, z_axis):
            orientation = [0, 0, 0, 1]
        elif np.allclose(bone_unit, -z_axis):
            orientation = [1, 0, 0, 0]
        else:
            cross = np.cross(z_axis, bone_unit)
            dot = np.dot(z_axis, bone_unit)
            angle = np.arccos(np.clip(dot, -1, 1))
            
            if np.linalg.norm(cross) > 1e-6:
                axis = cross / np.linalg.norm(cross)
                s = np.sin(angle / 2)
                orientation = [axis[0] * s, axis[1] * s, axis[2] * s, np.cos(angle / 2)]
            else:
                orientation = [0, 0, 0, 1]
        
        # 创建可视化圆柱体
        visual_shape = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=radius,
            length=bone_length,
            rgbaColor=color
        )
        
        # 创建圆柱体（无碰撞，仅可视化）
        cylinder_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=visual_shape,
            basePosition=center_pos,
            baseOrientation=orientation
        )
        
        return cylinder_id, visual_shape, bone_length
    
    def _update_bone_cylinder(self, cylinder_id, visual_shape, start_pos, end_pos, old_length, radius=0.02, color=[0, 1, 0, 0.8]):
        """更新现有骨骼圆柱体的位置和方向"""
        bone_vector = end_pos - start_pos
        bone_length = np.linalg.norm(bone_vector)
        
        if bone_length < 0.01:
            return old_length
        
        # 圆柱体中心位置
        center_pos = (start_pos + end_pos) / 2
        
        # 计算圆柱体方向
        bone_unit = bone_vector / bone_length
        z_axis = np.array([0, 0, 1])
        
        # 计算旋转四元数
        if np.allclose(bone_unit, z_axis):
            orientation = [0, 0, 0, 1]
        elif np.allclose(bone_unit, -z_axis):
            orientation = [1, 0, 0, 0]
        else:
            cross = np.cross(z_axis, bone_unit)
            dot = np.dot(z_axis, bone_unit)
            angle = np.arccos(np.clip(dot, -1, 1))
            
            if np.linalg.norm(cross) > 1e-6:
                axis = cross / np.linalg.norm(cross)
                s = np.sin(angle / 2)
                orientation = [axis[0] * s, axis[1] * s, axis[2] * s, np.cos(angle / 2)]
            else:
                orientation = [0, 0, 0, 1]
        
        # 只更新位置和方向，不重新创建
        p.resetBasePositionAndOrientation(cylinder_id, center_pos, orientation)
        
        # 如果长度变化较大，才重新创建视觉形状
        if abs(bone_length - old_length) > 0.05:
            new_visual_shape = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=radius,
                length=bone_length,
                rgbaColor=color
            )
            p.changeVisualShape(cylinder_id, -1, shapeIndex=new_visual_shape)
            return bone_length
        
        return old_length

    def set_human_model(self, pos_10):
        """绘制/更新人体骨骼模型（流畅连续版本）
        
        Args:
            pos_10: 人体十个关键点的3D坐标，形状为 (10, 3)
        """
        # 确保输入格式正确
        pos_10 = np.array(pos_10)
        if pos_10.shape != (10, 3):
            print(f"警告: pos_10形状应为(10, 3)，当前为{pos_10.shape}")
            return
        
        # 初始化骨骼数据结构
        if not hasattr(self, 'bone_cylinders'):
            self.bone_cylinders = []
            self.bone_visual_shapes = []
            self.bone_lengths = []
            
            # 首次创建所有骨骼
            for i, (start_idx, end_idx) in enumerate(self.bone_connections):
                if start_idx >= len(pos_10) or end_idx >= len(pos_10):
                    self.bone_cylinders.append(None)
                    self.bone_visual_shapes.append(None)
                    self.bone_lengths.append(0)
                    continue
                
                start_pos = pos_10[start_idx]
                end_pos = pos_10[end_idx]
                
                result = self._create_bone_cylinder(start_pos, end_pos)
                if result:
                    cylinder_id, visual_shape, bone_length = result
                    self.bone_cylinders.append(cylinder_id)
                    self.bone_visual_shapes.append(visual_shape)
                    self.bone_lengths.append(bone_length)
                else:
                    self.bone_cylinders.append(None)
                    self.bone_visual_shapes.append(None)
                    self.bone_lengths.append(0)
            
            print(f"首次创建了 {sum(1 for c in self.bone_cylinders if c is not None)} 根骨骼")
        else:
            # 更新现有骨骼位置
            for i, (start_idx, end_idx) in enumerate(self.bone_connections):
                if (i >= len(self.bone_cylinders) or 
                    self.bone_cylinders[i] is None or
                    start_idx >= len(pos_10) or 
                    end_idx >= len(pos_10)):
                    continue
                
                start_pos = pos_10[start_idx]
                end_pos = pos_10[end_idx]
                
                # 更新骨骼位置（不重新创建，避免闪烁）
                new_length = self._update_bone_cylinder(
                    self.bone_cylinders[i], 
                    self.bone_visual_shapes[i],
                    start_pos, 
                    end_pos, 
                    self.bone_lengths[i]
                )
                self.bone_lengths[i] = new_length

    def set_predict_human_model(self, pos_pred):
        """绘制/更新预测人体骨骼模型（独立于当前人体模型）
        
        Args:
            pos_pred: 人体十个关键点的3D坐标，形状为 (N, 10, 3)
        """
        # 确保输入格式正确
        pos_pred = np.array(pos_pred)
        if pos_pred.ndim != 3 or pos_pred.shape[1:] != (10, 3):
            print(f"警告: pos_pred形状应为(N, 10, 3)，当前为{pos_pred.shape}")
            return
        
        # 初始化预测骨骼数据结构（独立于当前人体模型）
        if not hasattr(self, 'pred_bone_cylinders'):
            self.pred_bone_cylinders = []
            self.pred_bone_visual_shapes = []
            self.pred_bone_lengths = []
            self.pred_num_humans = 0
            
        # 清理旧的预测骨骼（如果预测人数发生变化）
        expected_bones = len(pos_pred) * len(self.bone_connections)
        if len(self.pred_bone_cylinders) != expected_bones:
            self._clear_prediction_bones()
            
            # 为每个预测的人创建骨骼
            for j in range(len(pos_pred)):
                pos_10 = pos_pred[j]
                for i, (start_idx, end_idx) in enumerate(self.bone_connections):
                    if start_idx >= len(pos_10) or end_idx >= len(pos_10):
                        self.pred_bone_cylinders.append(None)
                        self.pred_bone_visual_shapes.append(None)
                        self.pred_bone_lengths.append(0)
                        continue
                    
                    start_pos = pos_10[start_idx]
                    end_pos = pos_10[end_idx]
                    
                    # 为预测模型使用不同的颜色（蓝色/透明）
                    pred_color = [0, 0.5, 1, 0.6]  # 蓝色半透明
                    result = self._create_bone_cylinder(start_pos, end_pos, 
                                                      radius=0.015, color=pred_color)
                    if result:
                        cylinder_id, visual_shape, bone_length = result
                        self.pred_bone_cylinders.append(cylinder_id)
                        self.pred_bone_visual_shapes.append(visual_shape)
                        self.pred_bone_lengths.append(bone_length)
                    else:
                        self.pred_bone_cylinders.append(None)
                        self.pred_bone_visual_shapes.append(None)
                        self.pred_bone_lengths.append(0)
            
            self.pred_num_humans = len(pos_pred)
            print(f"创建了{len(pos_pred)}个预测人的{sum(1 for c in self.pred_bone_cylinders if c is not None)}根骨骼")
            
        else:
            # 更新现有预测骨骼位置
            bone_idx = 0
            for j in range(len(pos_pred)):
                pos_10 = pos_pred[j]
                for i, (start_idx, end_idx) in enumerate(self.bone_connections):
                    if (bone_idx >= len(self.pred_bone_cylinders) or 
                        self.pred_bone_cylinders[bone_idx] is None or
                        start_idx >= len(pos_10) or 
                        end_idx >= len(pos_10)):
                        bone_idx += 1
                        continue
                    
                    start_pos = pos_10[start_idx]
                    end_pos = pos_10[end_idx]
                    
                    # 更新预测骨骼位置
                    new_length = self._update_bone_cylinder(
                        self.pred_bone_cylinders[bone_idx], 
                        self.pred_bone_visual_shapes[bone_idx],
                        start_pos, 
                        end_pos, 
                        self.pred_bone_lengths[bone_idx],
                        radius=0.015,
                        color=[0, 0.5, 1, 0.6]
                    )
                    self.pred_bone_lengths[bone_idx] = new_length
                    bone_idx += 1

    def _clear_prediction_bones(self):
        """清理预测骨骼"""
        if hasattr(self, 'pred_bone_cylinders'):
            for cylinder_id in self.pred_bone_cylinders:
                if cylinder_id is not None:
                    try:
                        p.removeBody(cylinder_id)
                    except:
                        pass
            self.pred_bone_cylinders = []
            self.pred_bone_visual_shapes = []
            self.pred_bone_lengths = []
            self.pred_num_humans = 0

    def clear_prediction_display(self):
        """清理预测显示（外部调用接口）"""
        self._clear_prediction_bones()
        print("已清理预测显示")
    def step_simulation(self):
        """执行一步仿真"""
        p.stepSimulation()
    
    def disconnect(self):
        """断开PyBullet连接"""
        p.disconnect()






def test_robot_arm():
    """测试机械臂控制和人体骨骼动画的函数"""
    print("初始化机械臂可视化...")
    robot = RobotArmVisualizer()
    
    try:
        # 等待场景加载完成
        time.sleep(0.5)
        
        # 创建动态人体姿态序列
        def create_dynamic_pose_sequence():
            """创建流畅的人体动作序列"""
            poses = []
            num_frames = 120  # 2秒动画（60fps）
            
            for frame in range(num_frames):
                t = frame / num_frames * 2 * np.pi  # 完整周期
                
                # 基础姿态
                base_pose = np.array([
                    [0, 1, 1.7],    # 0: 头部
                    [0, 1, 1.4],    # 1: 颈部/肩部中心
                    [0, 1, 1.0],    # 2: 躯干中部
                    [0, 1, 0.5],    # 3: 臀部
                    [-0.2, 1, 1.3], # 4: 左肩
                    [-0.4, 1, 1.0], # 5: 左肘
                    [-0.6, 1, 0.7], # 6: 左手
                    [0.2, 1, 1.3],  # 7: 右肩
                    [0.4, 1, 1.0],  # 8: 右肘
                    [0.6, 1, 0.7],  # 9: 右手
                ])
                
                # 添加动态变化
                # 手臂摆动
                arm_swing = 0.3 * np.sin(t)
                base_pose[5][0] = -0.4 + arm_swing  # 左肘 x
                base_pose[6][0] = -0.6 + arm_swing * 1.5  # 左手 x
                base_pose[8][0] = 0.4 - arm_swing   # 右肘 x
                base_pose[9][0] = 0.6 - arm_swing * 1.5   # 右手 x
                
                # 垂直摆动
                vertical_wave = 0.1 * np.sin(t * 2)
                for i in range(len(base_pose)):
                    base_pose[i][2] += vertical_wave
                
                poses.append(base_pose.copy())
            
            return poses
        
        print("生成动态人体姿态序列...")
        pose_sequence = create_dynamic_pose_sequence()
        
        # 机械臂动作序列
        robot_angles_sequence = []
        num_frames = len(pose_sequence)
        for frame in range(num_frames):
            t = frame / num_frames * 2 * np.pi
            angles = [
                0.3 * np.sin(t),         # J1
                0.3 + 0.2 * np.cos(t),   # J2
                -0.2 * np.sin(t * 1.5),  # J3
                0.4 * np.cos(t * 0.8),   # J4
                0,                       # J5
                0,                       # J6
                0                        # J7
            ]
            robot_angles_sequence.append(angles)
        
        print(f"开始流畅动画演示 ({num_frames} 帧)...")
        print("机械臂和人体骨骼将同时运动")
        print("按 Ctrl+C 停止动画")
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            current_frame = frame_count % num_frames
            
            # 更新机械臂姿态
            robot.set_joint_angles(robot_angles_sequence[current_frame])
            
            # 更新人体骨骼（流畅更新，不重新创建）
            robot.set_human_model(pose_sequence[current_frame])
            
            # 执行仿真步进（提高帧率）
            robot.step_simulation()
            
            # 控制帧率到60fps
            time.sleep(1/60)
            
            frame_count += 1
            
            # 每60帧打印一次性能信息
            if frame_count % 60 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                print(f"帧 {frame_count}, 实际FPS: {fps:.1f}")
    
    except KeyboardInterrupt:
        print("\n用户中断动画")
    
    finally:
        print("关闭可视化...")
        robot.disconnect()


def test_simple_poses():
    """测试简单静态姿态"""
    print("初始化机械臂可视化...")
    robot = RobotArmVisualizer()
    
    try:
        time.sleep(0.5)
        
        # 测试静态姿态
        test_poses = [
            # 标准站立
            np.array([
                [0, 1, 1.7], [0, 1, 1.4], [0, 1, 1.0], [0, 1, 0.5],
                [-0.2, 1, 1.3], [-0.4, 1, 1.0], [-0.6, 1, 0.7],
                [0.2, 1, 1.3], [0.4, 1, 1.0], [0.6, 1, 0.7]
            ]),
            # 伸展姿势
            np.array([
                [0, 1, 1.7], [0, 1, 1.4], [0, 1, 1.0], [0, 1, 0.5],
                [-0.3, 1, 1.3], [-0.7, 1, 1.3], [-1.0, 1, 1.3],
                [0.3, 1, 1.3], [0.7, 1, 1.3], [1.0, 1, 1.3]
            ])
        ]
        
        test_angles = [
            [0, 0.3, 0, 0, 0, 0, 0],
            [0.5, 0.3, -0.2, 0.4, 0, 0, 0]
        ]
        
        for i, (pose, angles) in enumerate(zip(test_poses, test_angles)):
            print(f"\n=== 静态测试 {i+1} ===")
            robot.set_joint_angles(angles)
            robot.set_human_model(pose)
            robot.step_simulation()
            print("按回车继续...")
            input()
    
    except KeyboardInterrupt:
        print("用户中断")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    print("选择测试模式:")
    print("1. 流畅动画演示 (推荐)")
    print("2. 静态姿态测试")
    
    choice = input("请输入选择 (1 或 2, 默认1): ").strip()
    
    if choice == "2":
        test_simple_poses()
    else:
        test_robot_arm()
