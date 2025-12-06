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

def GetTranScope(a ,alpha ,d ,theta ,yeta):
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
    T =  Tz @ Rz @ Tx @ Rx
    return T

def GetFKinematics(theta, din = 0):
    T01 = GetTran2(0, 0, 0.2405, 0, float(theta[0]))
    T12 = GetTran2(0, -pi/2, 0, 0, float(theta[1]))
    T23 = GetTran2(0, pi/2, 0.256, 0, float(theta[2]))
    T34 = GetTran2(0, -pi/2, 0, 0, float(theta[3]))
    T45 = GetTran2(0, pi/2, 0.21, 0, float(theta[4]))
    T56 = GetTran2(0, -pi/2, 0, 0, float(theta[5]))
    T67 = GetTran2(0, pi/2, 0.144, 0, float(theta[6]))
    T77 = GetTranScope(0.23 - din, 0, 0.18, -pi/6, 0)
    T = T01 @ T12 @ T23 @ T34 @ T45 @ T56 @ T67 @ T77
    return T, [T01, T12, T23, T34, T45, T56, T67, T77]

def GetRPY(theta, din = 0):
    T, _ = GetFKinematics(theta, din)
    R = T[:3,:3]
    roll = np.arctan2(R[2, 1], R[2, 2])
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw
 
def GetJacob(theta, din = 0):
    """
    计算末端坐标系雅可比矩阵
    """
    # 齐次变换矩阵计算 ij代表：在i坐标系下看j
    _, T = GetFKinematics(theta, din)
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

def GetBase2JointVel(vend, theta, din = 0):
    """
    将 base下机械臂末端速度 转换为 关节角速度
    vend 6*1 [w,v].T
    """
    T, _ = GetFKinematics(theta, din)
    # 求该位姿下 物体 雅可比矩阵 与 雅可比的逆矩阵
    Jacob = GetJacob(theta, din)
    Jacob_n = np.linalg.pinv(Jacob)

    # 求在末端坐标系下速度
    R = np.eye(6)
    R[0:3, 0:3] = T[0:3, 0:3]
    R[3:6, 3:6] = T[0:3, 0:3]

    vend_ob = R.T @ vend
    # 计算各关节角速度
    v_theta = Jacob_n @ vend_ob
    return v_theta


def GetEnd2JointVel(vend, theta, din = 0):
    """
    将 End下机械臂末端速度 转换为 关节角速度
    vend 6*1 [w,v].T
    """
    # 求该位姿下 物体 雅可比矩阵 与 雅可比的逆矩阵
    Jacob = GetJacob(theta, din)
    Jacob_n = np.linalg.pinv(Jacob)

    # 计算各关节角速度
    v_theta = Jacob_n @ vend
    return v_theta


def GetEndVel(v_theta, theta, din = 0):
    """
    由关节角速度获得 在base下的结构末端速度
    theta 6*1
    """
    T, _ = GetFKinematics(theta, din)
    # 求该位姿下 物体 雅可比矩阵
    Jacob = GetJacob(theta, din)
    # 末端坐标系 下 速度
    vend_ob = Jacob @ v_theta
    # 坐标系转换
    R = np.eye(6)
    R[0:3, 0:3] = T[0:3, 0:3]
    R[3:6, 3:6] = T[0:3, 0:3]
    # base 下速度
    vend = R @ vend_ob

    return vend


def GetIKinematics(Td, theta0, din = 0):
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
        T06, _ = GetFKinematics(theta, din)
        dp = 1.0* (Td[0:3,3] - T06[0:3,3])
        dR = -0.5*(so3(Td[0:3,0]) @ T06[0:3,0] + so3(Td[0:3,1]) @ T06[0:3,1] + so3(Td[0:3,2]) @ T06[0:3,2])
        dA = np.concatenate((dR.reshape((3, 1)),dp.reshape((3, 1))), axis=0)
        dq = GetBase2JointVel(dA, theta, din)
        theta = theta + dq.flatten()
        delatQ = np.linalg.norm(dq[0:3,0])*10 + np.linalg.norm(dq[3:6,0])
        lmax = lmax +1
        if lmax > l_limit:
            LimitFlag = 1
            print('Solution wouldn''t converge')
    return theta

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
