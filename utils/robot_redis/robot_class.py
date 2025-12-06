from math import sin, cos, tan, atan, pi

import numpy as np
import matplotlib.pyplot as plt

class RobotControl:
    q1 = 0
    q2 = 0
    q3 = 0
    q4 = 0
    q5 = 0
    q_pos=[q1, q2, q3, q4, q5]
    d_in = 0
    T05=np.eye(4)
    T5rcm=np.zeros((4,4))
    test_mode= False
    T_stack = []
    Jacobian = np.zeros((6,5))
    dt = 0.01
    
    T0rcm=np.zeros((4,4))
    _integrate = 0
    
    def __init__(self, q_init=[0,0,0,0,0], gamma=0.5, testmode=False):
        """
        Initialize the class
        q_init:
            numbers(list): Initial positions of all joints
        gamma:
            float: Ratio of the scope that is inside human body
        testmode:
            boolean: Determines if the object will be constructed in test mode. In test mode, yeta1 and yeta2 will not be converted to x
        """
        self.q1=q_init[0]
        self.q2=q_init[1]
        self.q3=q_init[2]
        self.q4=q_init[3]
        self.q5=q_init[4]
        self.q_pos = [self.q1, self.q2, self.q3, self.q4, self.q5]
        self.d_in = 0.1945*gamma
        self.T5rcm=np.matrix([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., -self.d_in], [0., 0., 0., 1.]])
        self.test_mode=testmode
        self.setFKinematics()
        self.T0rcm=self.T05*self.T5rcm
        self.setJacob()

    # from robot_new_Jacob.py 
    def calc_Trans(self, a ,alpha ,d ,theta ,yeta):
        """
        Obtain the forward transformarion matrix (MDH representation): T = R(alpha_i-1)*T(a_i-1)*R(theta_i)*T(d_i)
        a:
            float: link length
        alpha:
            float: link twist angle
        d:
            joint offset
        theta:
            fixed joint angle offset
        yeta:
            joint angle
        """
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
    

    def so3(self, z):
        """
        Calculate skew symmetric metric of a vector
        z:
            np.array or np.matrix: a vector of 3 elements
        """
        # 叉乘
        z_so3 = np.matrix([[0., -float(z[2].item()), float(z[1].item())], [float(z[2].item()), 0., -float(z[0].item())], [-float(z[1].item()), float(z[0].item()), 0.]])
        return z_so3
    
    
    def Get2X(self):
        """
        计算从(x1 x2 x3 x4 x5关于时间导数) 转换为 (x1 yeta1 x3 yeta2 x5关于时间的导数) 的 转换矩阵
        X:
            numbers(list): joint positions
        """
        x1 = self.q1
        x2 = self.q2
        x3 = self.q3
        x4 = self.q4
        
        a = 1/(0.055 + (x1 - x2)*(x1 - x2)/0.055)
        b = 1/(0.055 + (x3 - x4)*(x3 - x4)/0.055)
        Tx = np.matrix([[1, 0, 0, 0, 0],[-a, a, 0, 0, 0],[0, 0, 1, 0, 0],[0, 0, b, -b, 0], [0, 0, 0, 0, 1]])
        return Tx
    
    
    def updateState(self, X, v_z):
        """
        Update joint positions, forward kinematics and spatial Jacobian
        X:
            numbers(list): joint positions
        """
        self.q1=X[0]
        self.q2=X[1]
        self.q3=X[2]
        self.q4=X[3]
        self.q5=X[4]
        self.q_pos = [self.q1, self.q2, self.q3, self.q4, self.q5]
        self.setFKinematics()
        self.setJacob()
        self.Trcm5=np.linalg.inv(self.T05)*self.T0rcm
        Prcm5 = self.Trcm5[0:3,[3]]
        self.d_in = np.linalg.norm(Prcm5)
        # self.d_in = self.d_in + v_z*self.dt


    # X is the array of joint positions, these values should come from motor readings
    def setFKinematics(self):
        x1 = self.q1
        x2 = self.q2
        x3 = self.q3
        x4 = self.q4
        x5 = self.q5
        if self.test_mode:
            yeta1 = x2
            yeta2 = x4
        else:
            yeta1 = atan((x2-x1)/0.055)
            yeta2 = atan((x3-x4)/0.055)
        # 齐次变换矩阵计算 ij代表：在i坐标系下看j1 
        
        T01 = self.calc_Trans(0, 0, 32e-3+x1, 0, 0) # Arrangement of variables: calc_Trans(a ,alpha ,d ,theta ,yeta)
        T12 = self.calc_Trans(0, -pi / 2, 0, -pi / 2, yeta1)
        T23 = self.calc_Trans(93e-3, 0, x3, pi / 2, 0)
        T34 = self.calc_Trans(0, pi / 2, 56.5e-3, pi/2, yeta2)
        # Frame 5 is at the tool tip
        T45 = self.calc_Trans(0, pi/2, 0.1945+x5, 0, 0)

        T05 = T01 * T12 * T23 * T34 * T45
        T = [T01, T12, T23, T34, T45]
        self.T05 = T05
        self.T_stack = T
    
    
    # Calculate Spatial Jacobian
    def setJacob(self):        
        T0i = np.eye(4)
        Tx=self.Get2X()
        Jacobian = []
        P05 = self.T05[0:3,[3]] # Position of end effector
        for i in range(5):
            T0i = T0i*self.T_stack[i]
            P0i = T0i[0:3,[3]] # Position of the ith joint
            z_i = T0i[0:3,[2]] # Orientation of the ith joint axis
            if i==1 or i==3:
                Jvi = self.so3(z_i)*(P05-P0i)
                Jwi = z_i
            else:
                Jvi = z_i
                Jwi = np.zeros((3,1))
            J_i=np.vstack((Jvi, Jwi))
            Jacobian.append(J_i)
        
        Jacob = np.hstack((Jacobian[0], Jacobian[1], Jacobian[2], Jacobian[3], Jacobian[4]))
        if self.test_mode:
            self.Jacobian = Jacob
        else:
            self.Jacobian = Jacob*Tx

    # Calculate the end velocity with the rcm constraints
    def getRCMEndvelocity(self, v_scope):
        '''
        : param v_scpoe: velocity of end effector without rcm constraints. v_scope=[wx,wy,wz,vz] 
        '''
        Trcm5 = np.linalg.inv(self.T0rcm)*self.T05 # transformation from rcm to 5
        
        
        adT0rcm=np.zeros((6,6)) # Adjoint transformation from 0 to rcm
        adT0rcm[0:3,0:3]=self.T0rcm[0:3, 0:3]
        adT0rcm[3:6,3:6]=self.T0rcm[0:3, 0:3]

        # R0rcm = self.T0rcm[0:3, 0:3]
        # P0rcm = self.T0rcm[0:3,[3]]
        # adT0rcm[3:6,0:3]=self.so3(P0rcm)*R0rcm
        
        adT05=np.zeros((6,6))
        P05=self.T05[0:3,[3]]
        adT05[0:3,0:3]=self.T05[0:3, 0:3]
        adT05[3:6,3:6]=self.T05[0:3, 0:3]
        # adT05[3:6,0:3]=self.so3(P05)*self.T05[0:3, 0:3]
        
        Rrcm5 = Trcm5[0:3, 0:3]
        Prcm5 = Trcm5[0:3,[3]]
        
        if v_scope[3] == 0:
            wrcm5=Rrcm5*np.matrix(v_scope[0:3]).T 
            vrcm5=np.cross(wrcm5.T, Prcm5.T)
        
            twistrcm5=np.vstack((wrcm5, vrcm5.T))
            twist05 = adT0rcm*twistrcm5
            w05 = twist05[0:3]
            v05 = twist05[3:6]
            
            v_e = np.vstack((v05, w05))

        else:
            twist5=np.matrix([0, 0, 0, 0, 0, v_scope[3]]).T
            twist05 = adT05*twist5
            w05 = twist05[0:3]
            v05 = twist05[3:6]
            
            v_e = np.vstack((v05, w05))

        # v_e = v_ev + v_ew
        v_cps = self.velocityCompensate()
        for i in range(3):
            v_e[i] += v_cps[i]
        return v_e
    
    # Convert velocity of end effector to joint velocities
    def getJointVelocity(self, v_scope):
        vend=self.getRCMEndvelocity(v_scope)
        return np.linalg.pinv(self.Jacobian)*vend
    
    def velocityCompensate(self):
        P0rcm = self.T0rcm[0:3,[3]]
        T5r=np.matrix([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., -self.d_in], [0., 0., 0., 1.]])
        T0r=self.T05*T5r
        P0r = T0r[0:3,[3]]
        delta_p=P0rcm-P0r

        v_com = 18*delta_p + 8*self._integrate
        self._integrate += delta_p

        return v_com

    
if __name__ == '__main__':
    q = [0., 0., 0., 0.0, 0.]
    # q = [0.0425, 0, 0, 0, 0.0330] 
    r=RobotControl(q_init=q, gamma=0.26,testmode=True)
    v_scope = [-0.4, 0, 0, 0]
    traj_p5=[]
    traj_rcm=[]
    traj_vel=[]
    
    for i in range(1000):
        if i == 200:
            v_scope=[0, 0.4, 0, 0]
        elif i == 400:
            v_scope=[0, 0, 0, 0.05]
        elif i == 600:
            v_scope=[0, 0, 0, -0.04]
        elif i == 800:
            v_scope=[0.4, 0, 0, 0]
        
        q_dot = r.getJointVelocity(v_scope)
        for j in range(5):
            q[j] += q_dot[j].item()*r.dt
        traj_vel.append(q_dot)
        
        r.updateState(q, v_scope[3])
        T5r=np.matrix([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., -r.d_in], [0., 0., 0., 1.]])
        T0r=r.T05*T5r
        Prcm = T0r[0:3,[3]] # position of the point on cylinder of the scope, which is d_in meter from the end of the scope.
        traj_rcm.append(Prcm) # The position of this point should be always conincidence with the rcm point
        P5=r.T05[0:3,[3]]

        traj_p5.append(P5)

    x = np.arange(0, 10, 0.01)
    traj_rcm = np.hstack(traj_rcm)
    traj_p5 = np.hstack(traj_p5)
    traj_vel = np.hstack(traj_vel)
    
    # plt.plot(x, traj_rcm.T)
    plt.plot(x, traj_vel.T)
    # plt.legend()
    plt.show()
