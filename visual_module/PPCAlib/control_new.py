import numpy as np
from .Kinematics import GetFKinematics, GetReduJointVel
from numpy import pi

class ControlSystem:
    """
    双层碰撞检测与避让控制系统
    
    第一层: 电流检测（实时物理碰撞，最高优先级）
    第二层: 视觉检测（预测性避障，次级优先级）
    
    状态机优先级（从高到低）:
    1. 电流异常 -> 紧急避让（忽略视觉）
    2. 视觉碰撞 -> 紧急避让
    3. 危险区 -> 缓慢避让
    4. 警告区 -> 保持位置
    5. 安全边缘 -> 慢速回归
    6. 安全区 -> 快速回归
    7. 已回归 -> 静止
    """
    
    def __init__(self):
        # ===== 关节状态 =====
        self.theta = np.zeros(7)
        self.theta_ori = np.zeros(7)  # 初始位置
        self.v_joint = np.zeros(7)
        self.index = 4  # 冗余关节索引
        self.dt = 0.01
        
        # ===== 速度控制 =====
        self.vavoid_ori = np.array([0., 0., 0., 0., 0., -0.3])
        self.vavoid = np.zeros(6)
        self.last_direction = 1  # 缓存上一次的避让方向
        
        # ===== 关节限位 =====
        self.thetalimit1 = np.deg2rad([-170, -100, -150, -130, -150, -120, -360])
        self.thetalimit2 = np.deg2rad([170, 100, 150, 130, 150, 120, 360])
        
        # ===== 电流检测参数（第一层防护）=====
        self.current_threshold = 400          # 电流阈值(mA)
        self.current_deadzone = 50            # 电流死区，避免抖动
        self.current_change_threshold = 200   # 电流变化率阈值(mA/cycle)
        self.last_current = 0                 # 上一次电流值
        self.in_emergency_mode = False        # 紧急模式标志
        self.emergency_duration = 0           # 紧急模式持续计数
        self.emergency_max_duration = 50      # 紧急模式最大持续周期（0.5秒）
        self.emergency_direction = 1          # 紧急避让方向
        
        # ===== 视觉检测参数（第二层防护）=====
        self.k_current_avoid = 100.0      # 电流避让增益（最高）
        self.k_collision = 50.0           # 视觉碰撞增益
        self.k_danger = 1.0               # 危险区增益
        self.k_return_slow = 10.0         # 慢速回归增益
        self.k_return_fast = 20.0         # 快速回归增益
        
        # ===== 距离阈值 =====
        self.d_danger = 0.1               # 危险距离
        self.d_warning = 0.15             # 警告距离
        self.d_safe = 0.2                 # 安全距离
        
        # ===== 回归判断阈值 =====
        self.theta_tolerance = 0.3        # 关节角度容差（弧度）
        
        # ===== 调试模式 =====
        self.debug_mode = True            # 是否打印调试信息

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
    
    def _check_current_collision(self, current):
        """
        检查电流是否异常（第一层防护）
        
        参数:
        - current: 当前关节电流值(mA)
        
        返回:
        - is_collision: 是否检测到碰撞
        - direction: 避让方向 (1 或 -1)
        """
        # 计算电流变化率
        dI = current - self.last_current
        
        # 检查绝对电流值
        abs_current = abs(current)
        current_exceeds = abs_current > self.current_threshold
        
        # 检查电流变化率
        sudden_change = abs(dI) > self.current_change_threshold
        
        # 电流碰撞判断
        is_current_collision = current_exceeds or sudden_change
        
        if is_current_collision:
            # 确定避让方向：电流正->反向避让，电流负->正向避让
            if current > self.current_deadzone:
                direction = -1  # 反向避让
            elif current < -self.current_deadzone:
                direction = 1   # 正向避让
            else:
                # 电流在死区内，使用变化率判断
                direction = -1 if dI > 0 else 1
            
            # 进入或维持紧急模式
            if not self.in_emergency_mode:
                self.in_emergency_mode = True
                self.emergency_duration = 0
                self.emergency_direction = direction
                if self.debug_mode:
                    print(f"[电流检测] 检测到碰撞！电流={current:.1f}mA, dI={dI:.1f}mA/cycle")
            
            self.emergency_duration += 1
            
        else:
            # 电流正常，检查是否退出紧急模式
            if self.in_emergency_mode:
                self.emergency_duration += 1
                if self.emergency_duration > self.emergency_max_duration:
                    self.in_emergency_mode = False
                    self.emergency_duration = 0
                    if self.debug_mode:
                        print(f"[电流检测] 退出紧急模式")
                direction = self.emergency_direction
            else:
                direction = 1
        
        # 更新上一次电流
        self.last_current = current
        
        return self.in_emergency_mode, direction
    
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
    
    def update(self, collision_detector, theta, current=0):
        """
        主控制更新函数 - 双层防护系统
        
        参数:
        - collision_detector: 碰撞检测结果字典（视觉检测）
        - theta: 当前关节角度
        - current: 当前关节电流(mA)，默认0表示不使用电流检测
        
        返回:
        - 更新后的关节角度
        """
        self.theta = np.array(theta)
        self.vavoid = np.zeros(6)
        self.vjoint = np.zeros(7)
        min_d = collision_detector.get('min_distance', float('inf'))
        dir = 1
        
        # ========== 第一层防护：电流检测（最高优先级）==========
        
        is_current_collision, current_direction = self._check_current_collision(current)
        
        if is_current_collision:
            # 电流异常，立即紧急避让，忽略所有视觉信息
            dir = current_direction
            # 使用最大增益进行避让
            scale = min(abs(current) / 100, 10)  # 根据电流大小调整速度
            self.vavoid = dir * self.vavoid_ori * self.k_current_avoid * scale
            self.vjoint = GetReduJointVel(self.vavoid, self.theta, self.index)
            
            if self.debug_mode:
                print(f"[状态0-电流] 物理碰撞！电流={current:.1f}mA, 方向={dir}, 速度缩放={scale:.2f}")
            
            # 跳过视觉检测逻辑，直接执行安全检查
            
        else:
            # ========== 第二层防护：视觉检测 ==========
            
            if collision_detector['is_collision']:
                # 状态1: 视觉碰撞 - 紧急避让
                dir = self._compute_avoidance_direction(collision_detector)
                self.vavoid = dir * self.vavoid_ori * self.k_collision
                self.vjoint = GetReduJointVel(self.vavoid, self.theta, self.index)
                if self.debug_mode:
                    print(f"[状态1-视觉] 碰撞！紧急避让, 方向={dir}")
                
            elif min_d < self.d_danger:
                # 状态2: 危险区 - 慢速避让
                dir = self._compute_avoidance_direction(collision_detector)
                # 距离越近，速度越大（反比例）
                min_d_safe = max(min_d, 0.03)  # 避免除零
                gain = self.k_danger / min_d_safe
                self.vavoid = dir * self.vavoid_ori * gain
                self.vjoint = GetReduJointVel(self.vavoid, self.theta, self.index)
                if self.debug_mode:
                    print(f"[状态2] 危险区！距离={min_d:.4f}m, 增益={gain:.2f}, 慢速避让")
                
            elif min_d < self.d_warning:
                # 状态3: 警告区 - 保持位置
                self.vjoint = np.zeros(7)
                if self.debug_mode:
                    print(f"[状态3] 警告区！距离={min_d:.4f}m, 保持位置")
                
            elif min_d < self.d_safe:
                # 状态4: 安全区边缘 - 慢速回归
                if not self._check_ori_theta(self.theta):
                    dir = self._compute_return_velocity()
                    self.vavoid = dir * self.vavoid_ori * self.k_return_slow
                    self.vjoint = GetReduJointVel(self.vavoid, self.theta, self.index)
                    if self.debug_mode:
                        print(f"[状态4] 安全区边缘！距离={min_d:.4f}m, 慢速回归")
                else:
                    self.vjoint = np.zeros(7)
                    if self.debug_mode:
                        print(f"[状态4] 已回归初始位置")
                    
            else:
                # 状态5: 安全区 - 快速回归
                if not self._check_ori_theta(self.theta):
                    dir = self._compute_return_velocity()
                    self.vavoid = dir * self.vavoid_ori * self.k_return_fast
                    self.vjoint = GetReduJointVel(self.vavoid, self.theta, self.index)
                    if self.debug_mode:
                        print(f"[状态5] 安全区！距离={min_d:.4f}m, 快速回归")
                else:
                    self.vjoint = np.zeros(7)
                    if self.debug_mode:
                        print(f"[状态5] 已回归初始位置，停止")
        
        # ========== 安全检查 ==========
        
        # 检查关节限位
        theta_next = self.theta + self.vjoint * self.dt
        if self._checktheta(theta_next):
            self.vjoint = np.zeros(7)
            if self.debug_mode:
                print("[安全] 关节限位！停止运动")
        
        # 更新关节角度
        self.theta = self.theta + self.vjoint * self.dt
        
        return self.theta
    
    def set_debug_mode(self, enable):
        """设置调试模式"""
        self.debug_mode = enable
    
    def get_status_info(self):
        """
        获取当前控制系统状态信息
        
        返回:
        - 状态信息字典
        """
        return {
            'in_emergency_mode': self.in_emergency_mode,
            'emergency_duration': self.emergency_duration,
            'last_current': self.last_current,
            'current_theta': self.theta.copy(),
            'theta_error': np.linalg.norm(self.theta - self.theta_ori),
            'is_at_origin': self._check_ori_theta(self.theta),
            'last_direction': self.last_direction
        }


# ========== 使用说明 ==========
"""
双层防护系统使用指南

初始化:
    controller = ControlSystem()
    controller.get_ori_joint(initial_theta)  # 设置初始位置
    controller.set_debug_mode(True)  # 可选：启用调试输出

主循环（带电流检测）:
    while True:
        # 获取当前电流（第一层防护）
        current = robot.get_joint_current()  # 单位：mA
        
        # 获取视觉检测结果（第二层防护）
        collision_info = collision_detector.detect(...)
        
        # 更新控制（电流优先）
        new_theta = controller.update(collision_info, current_theta, current)
        
        # 执行运动
        robot.move_to(new_theta)
        
        # 可选：获取状态信息
        status = controller.get_status_info()
        print(f"紧急模式: {status['in_emergency_mode']}")

主循环（仅视觉检测）:
    while True:
        collision_info = collision_detector.detect(...)
        new_theta = controller.update(collision_info, current_theta)  # current默认为0
        robot.move_to(new_theta)

collision_info 字典格式:
{
    'is_collision': bool,
    'min_distance': float,
    'collision_points': [CollisionPoint],  # CollisionPoint.normal 是法向量
    'nearest_points': [robot_point, human_point]  # 两个3D点
}

可调参数：

第一层防护（电流检测）:
- current_threshold: 电流阈值(mA)，默认400
- current_deadzone: 电流死区(mA)，默认50
- current_change_threshold: 电流变化率阈值(mA/cycle)，默认200
- k_current_avoid: 电流避让增益，默认100.0
- emergency_max_duration: 紧急模式最大持续周期，默认50

第二层防护（视觉检测）:
- k_collision: 视觉碰撞时的速度增益，默认50.0
- k_danger: 危险区速度增益，默认1.0
- k_return_slow/fast: 回归速度增益，默认10.0/20.0
- d_danger, d_warning, d_safe: 距离阈值，默认0.1/0.15/0.2

控制逻辑：
1. 优先检测电流，电流异常时忽略视觉信息
2. 电流正常时，使用视觉检测进行预测性避障
3. 多级距离阈值，实现渐进式控制
4. 安全回归机制，自动返回初始位置
"""
