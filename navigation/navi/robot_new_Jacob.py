import datetime
from math import pi
from math import sin,cos,atan,sqrt
import matplotlib.pyplot as plt
import numpy as np
from cvxopt import matrix, solvers
import random

global T
global T01, T12, T23, T34, T45, T05
T = []
T01 = T12 = T23 = T34 = T45 = T05 = np.eye(4)


def GetTran2(a ,alpha ,d ,theta ,yeta):
    # 沿Xi-1移动
    Tx = np.matrix([[1, 0, 0, a], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    # 绕Xi-1转动
    Rx = np.matrix([[1, 0, 0, 0], [0, cos(alpha), -1 * sin(alpha), 0], [0, sin(alpha), cos(alpha), 0], [0, 0, 0, 1]])
    # 沿Zi移动
    Tz = np.matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, d], [0, 0, 0, 1]])
    # 绕Zi转动
    theta = theta + yeta
    Rz = np.matrix([[cos(theta), -1 * sin(theta), 0, 0], [sin(theta), cos(theta), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    # 齐次变换矩阵
    T = Tx * Rx * Tz * Rz
    return T

def so3(z):
    # 叉乘
    z_so3 = np.matrix([[0., -float(z[2].item()), float(z[1].item())], [float(z[2].item()), 0., -float(z[0].item())], [-float(z[1].item()), float(z[0].item()), 0.]])
    return z_so3

# 从(x1 x2 x3 x4 关于时间导数) 转换为 (x1 yeta1 x3 yeta2 关于时间的导数) 的 转换矩阵
def Get2X(X):
    x1 = X[0]
    x2 = X[1]
    x3 = X[2]
    x4 = X[3]
    a = 1/(0.055 + (x1 - x2)*(x1 - x2)/0.055)
    b = 1/(0.055 + (x3 - x4)*(x3 - x4)/0.055)
    Tx = np.matrix([[1, 0, 0, 0],[-a, a, 0, 0],[0, 0, 1, 0],[0, 0, b, -b]])
    return Tx

# 输入X为m量级
def GetFKinematics(X):
    x1 = X[0]
    x2 = X[1]
    x3 = X[2]
    x4 = X[3]
    yeta1 = atan((x2-x1)/0.055)
    yeta2 = atan((x3-x4)/0.055)
    # 齐次变换矩阵计算 ij代表：在i坐标系下看j1
    T01 = GetTran2(27.5e-3, 0, 8e-3+x1, 0, 0)
    T12 = GetTran2(0, -pi / 2, 0, -pi / 2, yeta1)
    T23 = GetTran2(93e-3, 0, x3, pi / 2, 0)
    T34 = GetTran2(0, pi / 2, 0, 0, yeta2)
    # 有镜
    T45 = GetTran2(45e-3, 0, 56.5e-3, 0, 0)

    # 无镜
    # T45 = GetTran2(0, 0, 0, 0, 0)
    T05 = T01 * T12 * T23 * T34 * T45
    T = [T01, T12, T23, T34, T45]
    return T05, T


# 计算末端坐标系雅可比矩阵
def GetJacob(X):
    T05, T = GetFKinematics(X)
    # print(T05)
    Jacob = [0]*6
    # 物体雅可比矩阵计算
    for i in range(1,5):
        T_ij = np.eye(4)
        for j in range(i,5):
            T_ij = T_ij*T[j] # T_ij
        p_ij = np.matrix([[T_ij[0,3]],[T_ij[1,3]],[T_ij[2,3]]])
        for k in range(3):
            if abs(p_ij[k]) < 1e-5:
                p_ij[k] = 0
        # 由ij向ji过度，计算物体雅可比矩阵
        p_ji = -T_ij[0:3,0:3].T * p_ij
        # 轴向量 即角速度正向
        z_ji = np.matrix([[T_ij[2, 0]], [T_ij[2, 1]], [T_ij[2, 2]]])
        for k in range(3):
            if abs(z_ji[k]) < 1e-10:
                z_ji[k] = 0
        # 分别针对移动副与转动副
        if i%2 != 0:
            v = z_ji
            Jacob[i] = np.matrix([[0],[0],[0],[float(v[0].item())],[float(v[1].item())],[float(v[2].item())]])
        else:
            z_ji_so3 = so3(z_ji)
            v = -z_ji_so3 * p_ji
            Jacob[i] = np.matrix([[z_ji[0, 0]], [z_ji[1, 0]], [z_ji[2, 0]], [float(v[0].item())],[float(v[1].item())],[float(v[2].item())]])
    # 雅可比矩阵合并与返回
    Jacobb = np.hstack((Jacob[1], Jacob[2], Jacob[3], Jacob[4]))

    Tx = Get2X(X)
    # print(Tx)
    return Jacobb*Tx


# 将 base下机械臂末端速度 转换为 关节角速度
# vend 6*1 np.matrix
def GetBase2JointVel(vend, X):
    T05, T = GetFKinematics(X)
    # 求该位姿下 物体 雅可比矩阵 与 雅可比的逆矩阵
    Jacob = GetJacob(X)
    Jacob_n = np.linalg.pinv(Jacob)
    # 求在末端坐标系下速度
    R = np.eye(6)
    R[0:3, 0:3] = T05[0:3, 0:3]
    R[3:6, 3:6] = T05[0:3, 0:3]

    vend_ob = R.T * vend

    # 计算各关节角速度
    theta = Jacob_n * vend_ob
    return theta

# 将 End下机械臂末端速度 转换为 关节角速度
# vend 6*1 np.matrix
def GetEnd2JointVel(vend, X):
    # 求该位姿下 物体 雅可比矩阵 与 雅可比的逆矩阵
    Jacob = GetJacob(X)
    Jacob_n = np.linalg.pinv(Jacob)

    # 计算各关节角速度
    theta = Jacob_n * vend
    return theta

# 由关节角速度获得 在base下的结构末端速度
# theta 4*1
def GetBaseVel(theta, X):
    T05, T = GetFKinematics(X)

    # 求该位姿下 物体 雅可比矩阵
    Jacob = GetJacob(X)
    # 末端坐标系 下 速度
    vend_ob = Jacob * theta
    # 坐标系转换
    R = np.eye(6)
    R[0:3, 0:3] = T05[0:3, 0:3]
    R[3:6, 3:6] = T05[0:3, 0:3]
    # base 下速度
    vend = R * vend_ob

    return vend

# 由关节角速度获得 在End下的结构末端速度
# theta 4*1
def GetEndVel(theta, X):

    # 求该位姿下 物体 雅可比矩阵
    Jacob = GetJacob(X)
    # 末端坐标系 下 速度
    vend_ob = Jacob * theta

    return vend_ob

# 迭代求解逆运动学
# Td:目标齐次变换矩阵 theta0:初始值关节角度(求解依赖初值设定)，需要np.matrix
def GetIKinematics(Td, X0):
    efs = 1e-10
    X = X0
    delatQ = 1
    lmax = 0
    l_limit = 2000
    LimitFlag = 0
    while delatQ > efs and LimitFlag==0 :
        T05, T = GetFKinematics(X)
        dp = Td[0:3,3] - T05[0:3,3]
        dR = -0.5*(so3(Td[0:3,0])*T05[0:3,0] + so3(Td[0:3,1])*T05[0:3,1] + so3(Td[0:3,2])*T05[0:3,2])
        # dR = np.matrix([float(dR[0]),float(dR[1]),float(dR[2])]).T
        dA = np.concatenate((dR.reshape((3, 1)),dp.reshape((3, 1))), axis=0)
        dA = np.matrix([float(dA[0]),float(dA[1]),float(dA[2]),float(dA[3]),float(dA[4]),float(dA[5])]).T
        dq = GetBase2JointVel(dA, X)
        for i in range(4):
            X[i] = X[i] + float(dq[i])
        delatQ = np.linalg.norm(dq[0:3,0])*10 + np.linalg.norm(dq[3:6,0])
        lmax = lmax +1
        if lmax > l_limit:
            LimitFlag = 1
            print('Solution wouldn''t converge')
    return X

# 通过二次规划方法 调库 解决速度转化的问题
# 输入：末端速度   输出：关节速度
def GetQP_End2JointVel(vend, X):
    dt = 0.01
    P = matrix([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
    q = matrix([0., 0., 0., 0.])
    Jacob = GetJacob(X)

    A = matrix(Jacob)
    b = matrix(vend)
    I1 = 1. * np.eye(4)
    I2 = - I1
    G = matrix(np.bmat('I1;I2'))

    X_max = matrix([0.031, 0.031, 0.022, 0.022])
    X_min = matrix([0., 0., -0.022 , -0.022])
    X_now = matrix([float(X[0]), float(X[1]), float(X[2]), float(X[3])])
    X1 = 1/dt * (X_max-X_now)
    X2 = 1/dt * (X_now-X_min)
    print(X1,X2)
    h = matrix(np.bmat('X1;X2'))
    '''
    print('P', np.shape(P))
    print('q', np.shape(q))
    print('G', np.shape(G))
    print('h', np.shape(h))
    print('A', np.shape(A))
    print('b', np.shape(b))
    '''
    sol = solvers.qp(P, q, G, h, A, b, kktsolver='ldl', options={'kktreg':1e-9})
    # speed = sol['x']
    # print(speed.T * P * speed)

    return sol['x']

# gama 置入深度所占比例   v_scopy 内窥镜末端速度 4*1 [wz, vx, vy, vz].T
def GetRCM_scp2end(gama, v_scope):
    d = 0.1945

    # RCM 约束
    v_end = [0.,0.,0.,0.,0.,0.]
    v_end[0] = float(v_scope[0].item())
    v_end[1] = float(-v_scope[3].item()/(gama*d))
    v_end[2] = float(v_scope[2].item()/(gama*d))

    v_end[3] = float(v_scope[1].item())
    v_end[4] = float(((gama-1)*v_scope[2].item())/(gama))
    v_end[5] = float(((gama-1)*v_scope[3].item())/(gama))
    # print(v_end[4]+(1-gama)*v_end[2]*d)
    
    # print(v_end[5]-(1-gama)*v_end[1]*d)

    return np.matrix(v_end).T

if __name__ == '__main__':
    
    '''
    # 验证雅可比矩阵、速度级逆运动学
    X = [0.00, 0.00, 0.00, 0.00]
    v = np.matrix([0., 0., 0., 0., 0.01, 0.01]).T
    v_joint = GetBase2JointVel(v, X)
    print(v_joint)
    print(GetEndVel(v_joint , X))
    '''

    
    # 试验RCM约束
    # 验证雅可比矩阵、速度级逆运动学
    X = [0.00, 0.00, 0.00, 0.00] # 四个关节长度
    v_s = np.matrix([0., 0., 0.0, 0.1]).T # 内镜坐标系下速度
    v_e = GetRCM_scp2end(0.5, v_s) # 获得持镜机构末端速度
    print("ve_",v_e)
    v_joint = GetEnd2JointVel(v_e, X) # 获得各关节移动速度
    print(v_joint)
    print(GetEndVel(v_joint , X))
    # 需要补充由关节移动速度切换电机转速
    

    # T05, T =GetFKinematics(X)
    # print(T05)
    # print(GetJacob(X))
    #X = [0.00, 0.00, 0.00, 0.00]
    #v = np.matrix([0.1,0.5,0.2,0.2,0.3,1.]).T
    # v_joint = GetBase2JointVel(v, X)
    #print(v)
    #v_joint2 = GetQP_End2JointVel(v, X)
    # print('1',v_joint)
    # print('2',v_joint2)
    # print(GetEndVel(v_joint , X))
    # print(GetEndVel(np.matrix(v_joint2), X))