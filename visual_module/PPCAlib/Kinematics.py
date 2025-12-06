#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from math import sin, cos, pi
import datetime
import time
"""
---25/08/26---
FK, Ik, Jacob, Redundant, all vel2joint test OK. 
Torque test are coming soon.
"""
def GetTran2(a ,alpha ,d ,theta ,yeta):
    # 沿Xi-1移动
    Tx = np.array([[1, 0, 0, a], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    # 绕Xi-1转动
    Rx = np.array([[1, 0, 0, 0], [0, cos(alpha), -1 * sin(alpha), 0], [0, sin(alpha), cos(alpha), 0], [0, 0, 0, 1]])
    # 沿Zi移动
    Tz = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, d], [0, 0, 0, 1]])
    # 绕Zi转动
    theta = theta + yeta
    Rz = np.array([[cos(theta), -1 * sin(theta), 0, 0], [sin(theta), cos(theta), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    # 齐次变换矩阵
    T = Tx @ Rx @ Tz @ Rz
    return T

def so3(z):
    # 叉乘
    z = z.flatten()
    z_so3 = np.array([
        [0., -z[2], z[1]], 
        [z[2], 0., -z[0]], 
        [-z[1], z[0], 0.]])
    return z_so3

def GetFKinematics(theta):
    T01 = GetTran2(0, 0, 0.2405, pi/2+pi/6, float(theta[0]))
    T12 = GetTran2(0, -pi/2, 0, 0, float(theta[1]))
    T23 = GetTran2(0, pi/2, 0.256, 0, float(theta[2]))
    T34 = GetTran2(0, -pi/2, 0, 0, float(theta[3]))
    T45 = GetTran2(0, pi/2, 0.21, 0, float(theta[4]))
    T56 = GetTran2(0, -pi/2, 0, 0, float(theta[5]))
    T67 = GetTran2(0, pi/2, 0.144, 0, float(theta[6]))
    T77 = GetTran2(0, 0, 0, 0, 0)
    T = T01 @ T12 @ T23 @ T34 @ T45 @ T56 @ T67 @ T77
    return T, [T01, T12, T23, T34, T45, T56, T67, T77]

def GetFKinematicsXarm(theta):
    T01 = GetTran2(0, 0, 0.17262, pi/2, float(theta[0]))
    T12 = GetTran2(0, pi/2, 0, 0, float(theta[1]))
    T23 = GetTran2(0, -pi/2, 0.44, 0, float(theta[2]))
    T34 = GetTran2(0, pi/2, 0.0, 0, float(theta[3]))
    T45 = GetTran2(0, -pi/2, 0.380, 0, float(theta[4]))
    T56 = GetTran2(0, pi/2, 0.0, 0, float(theta[5]))
    T67 = GetTran2(0, -pi/2, 0.1648, 0, float(theta[6]))
    T77 = GetTran2(0, 0, 0, 0, 0)
    T = T01 @ T12 @ T23 @ T34 @ T45 @ T56 @ T67 @ T77
    return T, [T01, T12, T23, T34, T45, T56, T67, T77]

def GetRPY(theta):
    T, _ = GetFKinematics(theta)
    R = T[:3,:3]
    roll = np.arctan2(R[2, 1], R[2, 2])
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw
 
def GetJacob(theta):
    """
    计算末端坐标系雅可比矩阵
    """
    # 齐次变换矩阵计算 ij代表：在i坐标系下看j
    _, T = GetFKinematics(theta)
    Jacob = [0]*8
    # 物体雅可比矩阵计算
    for i in range(1,8):
        T_ij = np.eye(4)
        for j in range(i,8):
            T_ij = T_ij @ T[j] # T_ij
        p_ij = np.array([[T_ij[0,3]],[T_ij[1,3]],[T_ij[2,3]]])
        for k in range(3):
            if abs(p_ij[k]) < 1e-5:
                p_ij[k] = 0
        # 由ij向ji过度，计算物体雅可比矩阵
        p_ji = -T_ij[0:3,0:3].T @ p_ij
        # 轴向量 即角速度正向
        z_ji = np.array([[T_ij[2, 0]], [T_ij[2, 1]], [T_ij[2, 2]]])
        for k in range(3):
            if abs(z_ji[k]) < 1e-10:
                z_ji[k] = 0
        # 分别针对移动副与转动副
        z_ji_so3 = so3(z_ji)
        v = -z_ji_so3 @ p_ji
        Jacob[i] = np.array([[z_ji[0, 0]], [z_ji[1, 0]], [z_ji[2, 0]], [v[0,0]], [v[1,0]], [v[2,0]]])
        # print(Jacob[i])
    # 雅可比矩阵合并与返回
    Jacobb = np.hstack((Jacob[1], Jacob[2], Jacob[3], Jacob[4], Jacob[5], Jacob[6], Jacob[7]))
    return Jacobb

def GetJacob_jointi(theta, index):
    """
    计算第i关节末端坐标系雅可比矩阵
    index in [1,7]是关节编号
    """
    _, T = GetFKinematics(theta)
    Jacob = [0]*8
    # 物体雅可比矩阵计算
    for i in range(1,index+1):
        # print(f"i:{i}, index:{index}")
        T_ij = np.eye(4)
        if i < index:
            for j in range(i, index):
                T_ij = T_ij @ T[j]
        elif i == index:
            T_ij = np.eye(4)

        p_ij = np.array([[T_ij[0,3]],[T_ij[1,3]],[T_ij[2,3]]])
        for k in range(3):
            if abs(p_ij[k]) < 1e-5:
                p_ij[k] = 0
        # 由ij向ji过度，计算物体雅可比矩阵
        p_ji = -T_ij[0:3,0:3].T @ p_ij
        # 轴向量 即角速度正向
        z_ji = np.array([[T_ij[2, 0]], [T_ij[2, 1]], [T_ij[2, 2]]])
        for k in range(3):
            if abs(z_ji[k]) < 1e-10:
                z_ji[k] = 0
        # 分别针对移动副与转动副
        z_ji_so3 = so3(z_ji)
        v = -z_ji_so3 @ p_ji
        Jacob[i] = np.array([[z_ji[0, 0]], [z_ji[1, 0]], [z_ji[2, 0]], [v[0,0]], [v[1,0]], [v[2,0]]])
        # print(Jacob[i])
    # 雅可比矩阵合并与返回
    for i in range(1,8):
        if i == 1:
            Jacobb = Jacob[1]
        elif i <= index:
            Jacobb = np.hstack((Jacobb, Jacob[i]))
        else:
            Jacobb = np.hstack((Jacobb, np.zeros((6, 1))))
    return Jacobb

def GetReduJointVel(vjoint, theta, index):
    """
    在机械臂末端不变的情况下,变换机械臂位姿
    多任务优先级冗余机械臂运动学规划
    
    参数:
    vjoint: 需要改变的目标关节速度或目标任务速度 (低优先级任务)
    theta: 当前关节角度
    index: 需要改变的关节索引 (高优先级任务)
    
    返回:
    dot_theta: 满足约束条件的关节角速度
    """
    
    # 高优先级任务：保持index关节末端位置姿态不变
    J_end = GetJacob(theta)  # 获取第index关节的雅可比矩阵
    
    # 计算J_end的零空间基
    # 使用奇异值分解(SVD)求零空间
    U, s, Vt = np.linalg.svd(J_end)
    
    # 找到奇异值接近零的位置
    tol = 1e-10
    rank = np.sum(s > tol)
    
    # 零空间的基向量是V的最后(n-rank)列
    n = J_end.shape[1]  # 关节数量
    if rank < n:
        N_end = Vt[rank:, :].T  # 零空间基矩阵
    else:
        # 如果J_end满秩，则没有零空间，无法同时满足两个任务
        print("Warning: J_end is full rank, no null space available")
        return np.zeros((1, n)).flatten()
    
    # 低优先级任务：实现目标速度vjoint
    # vjoint是末端任务速度(6x1)，如果是关节速度需要转换
    v_target = vjoint
    J_obs = GetJacob_jointi(theta, index)  # 关节雅可比矩阵
    
    # 构造投影到零空间的观测雅可比矩阵
    J_obs_prime = J_obs @ N_end
    
    # 求解系数alpha: J_obs_prime * alpha = v_target
    # 使用最小二乘法求解
    try:
        if J_obs_prime.shape[0] <= J_obs_prime.shape[1]:
            # 欠定系统，使用伪逆
            alpha = np.linalg.pinv(J_obs_prime) @ v_target
        else:
            # 超定系统，使用最小二乘法
            alpha = np.linalg.lstsq(J_obs_prime, v_target, rcond=None)[0]
        
        # 计算最终的关节速度
        dot_theta = N_end @ alpha
        
        # 验证高优先级约束是否满足
        constraint_error = J_end @ dot_theta
        if np.linalg.norm(constraint_error) > 1e-8:
            print(f"Warning: High priority constraint violation: {np.linalg.norm(constraint_error)}")
        
        return dot_theta.flatten()
        
    except np.linalg.LinAlgError:
        print("Error: Cannot solve for alpha, returning zero velocity")
        return np.zeros((1, n)).flatten()
    
    except Exception as e:
        print(f"Error in GetReduJointVel: {e}")
        return np.zeros((1, n)).flatten()

def GetBase2JointVel(vend, theta):
    """
    将 base下机械臂末端速度 转换为 关节角速度
    vend 6*1 [w,v].T
    """
    T, _ = GetFKinematics(theta)
    # 求该位姿下 物体 雅可比矩阵 与 雅可比的逆矩阵
    Jacob = GetJacob(theta)
    Jacob_n = np.linalg.pinv(Jacob)

    # 求在末端坐标系下速度
    R = np.eye(6)
    R[0:3, 0:3] = T[0:3, 0:3]
    R[3:6, 3:6] = T[0:3, 0:3]

    vend_ob = R.T @ vend

    # 计算各关节角速度
    v_theta = Jacob_n @ vend_ob
    return v_theta


def GetEnd2JointVel(vend, theta):
    """
    将 End下机械臂末端速度 转换为 关节角速度
    vend 6*1 [w,v].T
    """
    # 求该位姿下 物体 雅可比矩阵 与 雅可比的逆矩阵
    Jacob = GetJacob(theta)
    Jacob_n = np.linalg.pinv(Jacob)

    # 计算各关节角速度
    v_theta = Jacob_n @ vend
    return v_theta


def GetEnd2JointVel_i(vend, theta, index):
    """
    将 关节i速度 转换为 关节角速度
    vend 6*1 [w,v].T
    """
    # 求该位姿下 物体 雅可比矩阵 与 雅可比的逆矩阵
    Jacob = GetJacob_jointi(theta, index)
    Jacob_n = np.linalg.pinv(Jacob)

    # 计算各关节角速度
    v_theta = Jacob_n @ vend
    return v_theta


def GetEndVel(v_theta, theta):
    """
    由关节角速度获得 在base下的结构末端速度
    theta 6*1
    """
    T, _ = GetFKinematics(theta)
    # 求该位姿下 物体 雅可比矩阵
    Jacob = GetJacob(theta)
    # 末端坐标系 下 速度
    vend_ob = Jacob @ v_theta
    # 坐标系转换
    R = np.eye(6)
    R[0:3, 0:3] = T[0:3, 0:3]
    R[3:6, 3:6] = T[0:3, 0:3]
    # base 下速度
    vend = R @ vend_ob

    return vend


def GetIKinematics(Td, theta0):
    """
    迭代求解逆运动学
    Td:目标齐次变换矩阵 theta0:初始值关节角度(求解依赖初值设定),需要np.array
    """
    efs = 1e-3
    theta = theta0
    delatQ = 1
    lmax = 0
    l_limit = 2000
    LimitFlag = 0
    while delatQ > efs and LimitFlag==0 :
        T06, _ = GetFKinematics(theta)
        dp = 1.0* (Td[0:3,3] - T06[0:3,3])
        dR = -0.5*(so3(Td[0:3,0]) @ T06[0:3,0] + so3(Td[0:3,1]) @ T06[0:3,1] + so3(Td[0:3,2]) @ T06[0:3,2])
        dA = np.concatenate((dR.reshape((3, 1)),dp.reshape((3, 1))), axis=0)
        dq = GetBase2JointVel(dA, theta)
        theta = theta + dq.flatten()
        delatQ = np.linalg.norm(dq[0:3,0])*10 + np.linalg.norm(dq[3:6,0])
        lmax = lmax +1
        if lmax > l_limit:
            LimitFlag = 1
            print('Solution wouldn''t converge')
    return theta

def TorqueBase2joint(theta, Fend):
    T, _ = GetFKinematics(theta)
    R = T[:3, :3]
    Jacob = GetJacob(theta)
    RR = np.eye(6)
    RR[:3, :3] = R.T
    RR[3:, 3:] = R.T
    F = RR @ Fend
    return Jacob.T @ F

"""
Td = np.array([[ -1.00000000e+00,  -7.49879891e-33,   1.22464680e-16,   5.90279757e-17],
      [  7.49879891e-33,  -1.00000000e+00,   0.00000000e+00,   1.56754790e-17],
      [  1.22464680e-16,   0.00000000e+00,   1.00000000e+00,   8.50500000e-01],
      [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])
"""

"""
Td = GetFKinematics([-0.222/180*pi, 22.428/180*pi, 37.158/180*pi, 1.105/180*pi, 34.468/180*pi, 195.196/180*pi])
theta0 = np.array([0,0.17444,0.52333,0,0.35,3.14]).T
print(Td)
theta = GetIKinematics(Td, theta0)
print(GetFKinematics(theta))
print(theta)
"""

"""
print(GetQP_End2JointVel(np.array([0.,0.,0.,0.01,0.,0.]).T ,[-0.222/180*pi, 22.428/180*pi, 37.158/180*pi, 1.105/180*pi, 34.468/180*pi, 195.196/180*pi]))
print(GetEnd2JointVel(np.array([0.,0.,0.,0.01,0.,0.]).T ,[-0.222/180*pi, 22.428/180*pi, 37.158/180*pi, 1.105/180*pi, 34.468/180*pi, 195.196/180*pi]))
"""
"""
s_tips = np.array([220, 100]).T
A = GetQP_A_RCM(s_tips,[0.,0.,0.,0.,0.,0.])
A = array(A)
print(A)
"""
# 测试代码
if __name__ == "__main__":
    # 测试参数 - 使用非零关节角度
    theta_test = [0.1, 0.2, 0.3, 0.1, 0.2, 0.1, 0.1]
    vend = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    theta = [0. ,0. ,0., pi/2, 0., 0., 0.]
    time1 = time.time()
    T, _ = GetFKinematics(theta)
    print(T)
    # GetJacob(theta_test)
    # GetJacob_jointi(theta_test, 4)
    # GetBase2JointVel(vend, theta_test)
    rpy = GetRPY(theta)
    time2 = time.time()
    print(rpy)
    print(f"Forward kinematics computation time: {time2 - time1:.6f} seconds")

    print(2.3562/3.1415*180)