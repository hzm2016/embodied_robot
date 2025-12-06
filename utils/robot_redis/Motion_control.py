from math import sin, cos, tan, atan, pi

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
# from robot_new_Jacob.py 
# Obtain the forward transformarion matrix (MDH representation) R(alpha_i-1)*T(a_i-1)*R(theta_i)*T(d_i)
def calc_Trans(a ,alpha ,d ,theta ,yeta):
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
    T = Rx * Tx * Rz * Tz
    return T

# Obtain the skew-sytmmetric matrix of vector z, z is 3*1 vector
def so3(z):
    # 叉乘
    z_so3 = np.matrix([[0., -float(z[2].item()), float(z[1].item())], [float(z[2].item()), 0., -float(z[0].item())], [-float(z[1].item()), float(z[0].item()), 0.]])
    return z_so3

# 计算从(x1 x2 x3 x4 x5关于时间导数) 转换为 (x1 yeta1 x3 yeta2 x5关于时间的导数) 的 转换矩阵
def Get2X(X):
    x1 = X[0]
    x2 = X[1]
    x3 = X[2]
    x4 = X[3]
    
    a = 1/(0.055 + (x1 - x2)*(x1 - x2)/0.055)
    b = 1/(0.055 + (x3 - x4)*(x3 - x4)/0.055)
    Tx = np.matrix([[1, 0, 0, 0, 0],[-a, a, 0, 0, 0],[0, 0, 1, 0, 0],[0, 0, b, -b, 0], [0, 0, 0, 0, 1]])
    return Tx

# Calculate forward kinematics, X is the list joint positions
def GetFKinematics(X, testmode=False):
    x1 = X[0]
    x2 = X[1]
    x3 = X[2]
    x4 = X[3]
    x5 = X[4]
    if testmode:
        yeta1 = x2
        yeta2 = x4
    else:    
        yeta1 = atan((x2-x1)/0.055)
        yeta2 = atan((x3-x4)/0.055)
    # 齐次变换矩阵计算 ij代表：在i坐标系下看j1 
    T01 = calc_Trans(27.5e-3, 0, 8e-3+x1, 0, 0) # Arrangement of variables: calc_Trans(a ,alpha ,d ,theta ,yeta)
    T12 = calc_Trans(0, -pi / 2, 0, -pi / 2, yeta1)
    T23 = calc_Trans(93e-3, 0, x3, pi / 2, 0)
    T34 = calc_Trans(45e-3, pi / 2, 56.5e-3, pi/2, yeta2)
    # Frame 5 is at the tool tip
    T45 = calc_Trans(0, pi/2, 0.1945+x5, 0, 0)

    T05 = T01 * T12 * T23 * T34 * T45
    T = [T01, T12, T23, T34, T45]
    return T05, T

# Calculate Spatial Jacobian
def GetJacob(X,testmode=False):
    T05, T = GetFKinematics(X,testmode=testmode)
    T0i = np.eye(4)
    Tx=Get2X(X)
    Jacobian = []
    P05 = T05[0:3,[3]] # Position of end effector
    for i in range(5):
        T0i = T0i*T[i]
        P0i = T0i[0:3,[3]] # Position of the ith joint
        z_i = T0i[0:3,[2]] # Orientation of the ith joint axis
        if i==1 or i==3:
            Jvi = so3(z_i)*(P05-P0i)
            Jwi = z_i
        else:
            Jvi = z_i
            Jwi = np.zeros((3,1))
        J_i=np.vstack((Jvi, Jwi))
        Jacobian.append(J_i)
    
    Jacob = np.hstack((Jacobian[0], Jacobian[1], Jacobian[2], Jacobian[3], Jacobian[4]))
    if testmode:
        return Jacob
    else:
        return Jacob*Tx
    

# get the end effector velocity wrt base frame with rcm constraint, v_scope=[wx wy wz vz]
def getRCMEndVelocity(v_scope, X, testmode=False):
    T0rcm=np.matrix([[ 0., 0.,  1.00000000e+00,  2.02750000e-01],
                    [ 1.00000000e+00,  0., 0., 0.],
                    [ 0.,  1.00000000e+00,  0.,  2.00000000e-01],
                    [ 0.,  0.,  0.,  1.00000000e+00]]) # Position of RCM wrt the base frame is fixed, which will not change with the motion of the robot

    T05, T = GetFKinematics(X, testmode)
    
    Trcm5 = np.linalg.inv(T0rcm)*T05 # transformation from rcm to 5
    
    R=[]
    adT0rcm=np.zeros((6,6)) # Adjoint transformation from 0 to rcm, convert the velocities of the end effector wrt the rcm frame to velocities wrt the base frame
    adT0rcm[0:3,0:3]=T0rcm[0:3, 0:3]
    adT0rcm[3:6,3:6]=T0rcm[0:3, 0:3]
    adT05=np.zeros((6,6))
    adT05[0:3,0:3]=T05[0:3, 0:3]
    adT05[3:6,3:6]=T05[0:3, 0:3]
    for i in range(3):
        R.append(Trcm5[0:3,[i]])
    Rrcm5 = np.hstack((R[0],R[1],R[2]))
    
    Prcm5 = Trcm5[0:3,[3]]
    
    if v_scope[3] == 0:
        wrcm5=Rrcm5*np.matrix(v_scope[0:3]).T
        vrcm5=np.cross(wrcm5.T, Prcm5.T)
        
        twistrcm5=np.vstack((vrcm5.T, wrcm5))
        v_e = adT0rcm*twistrcm5
    else:
        twist5=np.matrix([0, 0, v_scope[3], 0, 0, 0]).T
        v_e = adT05*twist5
        
    return v_e

def getJointVelocity(X, v_scope, testmode):
    v_e = getRCMEndVelocity(v_scope, X, testmode)
    
    J = GetJacob(X, testmode)
    q_dot= np.linalg.pinv(J)*v_e

    return q_dot

def checkBoundry(X, q_dot, C, dt, r, testmode=False):
    center = np.matrix(C)
    for i in range(5):
        X[i] = q_dot[i]*dt
    T05, _ = GetFKinematics(X, testmode=testmode)
    P05 = T05[0:3,[3]]
    dist = np.linalg.norm(P05-center)

    return dist>r

if __name__ == '__main__':
    dt=0.1
    d_in=0.1945/2
    traj_q=[]
    traj_rcm=[]
    # q = [0.0425, 0, 0, 0, 0.0330] 
    q = [0.0425, 0.0425, 0., 0., 0.0330] # data for testing the functions, data from matlab 
    v_scope = [0, 0, 0, 0.08]
    
    for i in range(100):
        if i == 20:
            v_scope=[0, 0.2, 0, 0]
        elif i == 40:
            v_scope=[0, -0.2, 0, 0]
        elif i == 60:
            v_scope=[0, 0, 0, -0.04]
        elif i == 80:
            v_scope=[0.2, 0, 0, 0]
        traj_q.append(q)
        T05, _ = GetFKinematics(q,testmode=False)
        T5r=np.matrix([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., -d_in], [0., 0., 0., 1.]]) 
        T0r=T05*T5r
        Prcm = T0r[0:3,[3]] # position of the point on cylinder of the scope, which is d_in meter from the end of the scope.
        traj_rcm.append(Prcm) # The position of this point should be always conincidence with the rcm point
        q_dot = getJointVelocity(q, v_scope, testmode=False)
        
        for j in range(5):
            q[j] += q_dot[j].item()*dt
        d_in = d_in + v_scope[3]*dt
    traj_rcm=np.hstack(traj_rcm)
    traj_q=np.vstack(traj_q)
    x = np.arange(0,10,0.1)
    plt.plot(x,traj_rcm.T)
    plt.show()
    print(traj_rcm[:,[99]]-traj_rcm[:,[0]])