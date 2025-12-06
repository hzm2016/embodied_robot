import numpy as np
from .Kinematics import GetFKinematics, GetReduJointVel
from numpy import pi

class ControlSystem:
    def __init__(self):
        self.theta = np.zeros(7)
        self.theta_ori = np.zeros(7)  # 初始位置
        self.v_joint = np.zeros(7)
        self.index = 4  # 冗余关节索引
        self.dt = 0.01
        self.vavoid_ori = np.array([0., 0., 0., 0., 0., -0.7])
        self.vavoid = np.zeros(6)
        self.last_direction = 1  # 缓存上一次的避让方向
        # 关节限位
        self.thetalimit1 = np.deg2rad([-170, -100, -150, -130, -150, -120, -360])
        self.thetalimit2 = np.deg2rad([170, 100, 150, 130, 150, 120, 360])
        
        # 速度增益参数
        self.k_collision = 50.0      # 碰撞时的速度增益
        self.k_danger = 1.0          # 危险区速度增益
        self.k_return_slow = 10.0    # 慢速回归增益
        self.k_return_fast = 20.0    # 快速回归增益
        
        # 距离阈值
        self.d_danger = 0.1          # 危险距离阈值
        self.d_warning = 0.15        # 警告距离阈值
        self.d_safe = 0.2            # 安全距离阈值
        
        # 回归判断阈值
        self.theta_tolerance = 0.3   # 关节角度容差（弧度）

    def _checktheta(self, theta):
        """检查关节角度是否超出限位"""
        for i in range(len(theta)):
            if theta[i] < self.thetalimit1[i] or theta[i] > self.thetalimit2[i]:
                return True
        return False
    
    def _check_ori_theta(self, theta):
        """检查是否已回到初始位置"""
        return np.linalg.norm(theta - self.theta_ori) < self.theta_tolerance
    
    def get_ori_joint(self, theta):
        """设置初始关节角度"""
        self.theta_ori = np.array(theta)
    
    def _compute_avoidance_direction(self, collision_detector):
        """
        计算避让方向（笛卡尔空间）
        返回: 避让方向系数 (1 或 -1)
        """
        direction = 1
        if collision_detector['is_collision']:
            # 情况1: 发生碰撞，使用缓存的方向继续避让
            direction = self.last_direction
            print(f"[碰撞模式] 使用缓存方向: {direction}")
                
        elif 'nearest_points' in collision_detector and len(collision_detector['nearest_points']) == 2:
            # 情况2: 未碰撞，使用最近点计算方向并缓存
            robot_point = np.array(collision_detector['nearest_points'][0])  # 机械臂上的最近点
            human_point = np.array(collision_detector['nearest_points'][1])  # 人体上的最近点
            
            # 计算从人体最近点指向机械臂最近点的向量（避让方向）
            direction_ = robot_point - human_point
            x_axis = np.array([1, 0, 0])
            if np.dot(direction_, x_axis) < 0:
                direction = 1
            else:
                direction = -1
            # 缓存当前方向
            self.last_direction = direction
            print(f"[避让模式] 计算新方向并缓存: {direction}")
        else:
            # 情况3: 无碰撞信息，使用缓存方向
            direction = self.last_direction
            print(f"[默认模式] 使用缓存方向: {direction}")
            
        return direction
    
    def _compute_return_velocity(self):
        """
        计算回归初始位置的关节速度
        使用简单的比例控制
        """
        theta_error = self.theta_ori[2] - self.theta[2]
        if theta_error > 0:
            direction = 1
        else:
            direction = -1
        # 直接在关节空间规划，速度正比于误差
        return direction  # 返回关节误差，后续乘以增益
    
    def update(self, collision_detector, theta):
        """
        主控制更新函数
        
        参数:
        - collision_detector: 碰撞检测结果字典
        - theta: 当前关节角度
        
        返回:
        - 更新后的关节角度
        """
        self.theta = np.array(theta)
        self.vavoid = np.zeros(6)
        self.vjoint = np.zeros(7)
        min_d = collision_detector.get('min_distance', float('inf'))
        dir = 1
        # ========== 控制状态机 ==========
        
        if collision_detector['is_collision']:
            # 状态1: 碰撞 - 紧急避让
            dir = self._compute_avoidance_direction(collision_detector)
            self.vavoid = dir * self.vavoid_ori * self.k_collision
            self.vjoint = GetReduJointVel(self.vavoid, self.theta, self.index)
            print(f"[状态1] 碰撞！紧急避让")
            
        elif min_d < self.d_danger:
            # 状态2: 危险区 - 慢速避让
            dir = self._compute_avoidance_direction(collision_detector)
            # 距离越近，速度越大（反比例）
            min_d_safe = max(min_d, 0.03)  # 避免除零
            gain = self.k_danger / min_d_safe
            self.vavoid = dir * self.vavoid_ori * gain
            self.vjoint = GetReduJointVel(self.vavoid, self.theta, self.index)
            print(f"[状态2] 危险区！距离={min_d:.4f}m, 慢速避让")
            
        elif min_d < self.d_warning:
            # 状态3: 警告区 - 保持位置
            self.vjoint = np.zeros(7)
            print(f"[状态3] 警告区！距离={min_d:.4f}m, 保持位置")
            
        elif min_d < self.d_safe:
            # 状态4: 安全区边缘 - 慢速回归
            if not self._check_ori_theta(self.theta):
                dir = self._compute_return_velocity()
                self.vavoid = dir * self.vavoid_ori * self.k_return_slow
                self.vjoint = GetReduJointVel(self.vavoid, self.theta, self.index)
                print(f"[状态4] 安全区边缘！距离={min_d:.4f}m, 慢速回归")
            else:
                self.vjoint = np.zeros(7)
                print(f"[状态4] 已回归初始位置")
                
        else:
            # 状态5: 安全区 - 快速回归
            if not self._check_ori_theta(self.theta):
                dir = self._compute_return_velocity()
                self.vavoid = dir * self.vavoid_ori * self.k_return_fast
                self.vjoint = GetReduJointVel(self.vavoid, self.theta, self.index)
                print(f"[状态5] 安全区！距离={min_d:.4f}m, 快速回归")
            else:
                self.vjoint = np.zeros(7)
                print(f"[状态5] 已回归初始位置，停止")
        
        # ========== 安全检查 ==========
        
        # 检查关节限位
        theta_next = self.theta + self.vjoint * self.dt
        if self._checktheta(theta_next):
            self.vjoint = np.zeros(7)
            print("[安全] 关节限位！停止运动")
        
        # 更新关节角度
        self.theta = self.theta + self.vjoint * self.dt
        
        return self.theta


# ========== 使用说明 ==========
"""
初始化:
    controller = ControlSystem()
    controller.get_ori_joint(initial_theta)  # 设置初始位置

主循环:
    while True:
        collision_info = collision_detector.detect(...)
        new_theta = controller.update(collision_info, current_theta)
        robot.move_to(new_theta)

collision_info 字典格式:
{
    'is_collision': bool,
    'min_distance': float,
    'collision_points': [CollisionPoint],  # CollisionPoint.normal 是法向量
    'nearest_points': [robot_point, human_point]  # 两个3D点
}

可调参数:
- k_collision: 碰撞时的避让速度增益
- k_danger: 危险区速度增益
- k_return_slow/fast: 回归速度增益
- d_danger, d_warning, d_safe: 距离阈值
"""
