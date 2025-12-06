import queue
import socket
import numpy as np
import cv2
import matplotlib.pyplot as plt
import threading
import serial
from math import sin, cos, tan, atan, pi
import time

velocity_queue=queue.Queue()
joy_input = threading.Event()

class JoyReader(threading.Thread):
    def __init__(self, port='COM3', baudrate=9600, timeout=1):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.data = None
        self.running = False
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
        except serial.SerialException as e:
            print(f"Error opening serial port: {e}")
            self.ser = None

    def read_serial_data(self):
        if not self.ser:
            return

        try:
            line = self.ser.readline().decode('utf-8', errors='ignore').strip()
            
            if line:
                data = line.split(',')
                if len(data) != 5:
                    return
                
                data = [int(x) for x in data]
                data[2] = data[2] - 512
                data[3] = data[3] - 512

                if abs(data[2]) < 50:
                    data[2] = 0
                if abs(data[3]) < 50:
                    data[3] = 0
                if data[4]<50:
                    data[4] = 0
                else:
                    data[4] = 1
                self.data = data
                return data
            
        except ValueError as e:
            print(f"Error parsing line: {line} - {e}")
            return

    def run(self):
        if self.ser:
            self.running = True
            # print('Joyreader Thread waiting for reset to complete...')
            # reset_barrier.wait()
            print('Joyreader Thread started!')
            while self.running:
                v_scope=[0,0,0,0]
                try:
                    data = self.read_serial_data()
                    print(data)
                    if data[0] == 0:
                        v_scope = [0, 0, 0, 0.05]
                        velocity_queue.put(v_scope)
                    elif data[1] == 0:
                        v_scope = [0, 0, 0, -0.05]
                        velocity_queue.put(v_scope)
                    elif data[2] > 0 and abs(data[2]) > abs(data[3]):
                        v_scope = [0, 0.1, 0, 0]
                        velocity_queue.put(v_scope)
                    elif data[2] < 0 and abs(data[2]) > abs(data[3]):
                        v_scope = [0., -0.1, 0., 0.]
                        velocity_queue.put(v_scope)
                    elif data[3] > 0 and abs(data[3]) > abs(data[2]):
                        v_scope = [-0.1, 0., 0., 0.]
                        velocity_queue.put(v_scope)
                    elif data[3] < 0 and abs(data[3]) > abs(data[2]):
                        v_scope = [0.1, 0., 0., 0.]
                        velocity_queue.put(v_scope)
                    joy_input.set()
                except TypeError:
                    print("No input from joystick")
                
    def stop(self):
        self.running = False
        if self.ser:
            print('Joystick stopped.')
            self.ser.close()

class RobotControl(threading.Thread):
    dt = 0.01
    _integrate = 0
    
    def __init__(self, q_init=[0,0,0,0,0], gamma=0, testmode=False):
        """
        Initialize the class
        q_init:
            numbers(list): Initial positions of all joints
        gamma:
            float: Ratio of the scope that is inside human body
        testmode:
            boolean: Determines if the object will be constructed in test mode. In test mode, yeta1 and yeta2 will not be converted to x
        """
        super().__init__()
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
        self.running=True

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
        
        adT05=np.zeros((6,6))
        P05=self.T05[0:3,[3]]
        adT05[0:3,0:3]=self.T05[0:3, 0:3]
        adT05[3:6,3:6]=self.T05[0:3, 0:3]
        
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
    

    def run(self):
        # print('RobotControl Thread waiting for reset to complete...')
        # reset_barrier.wait()
        print('RobotControl Thread started!')
        
        while self.running:
            joy_input.wait()
            
            try:
                v_scope = velocity_queue.get()
                v_joint = self.getJointVelocity(v_scope)
                new_q1 = self.q1 + v_joint[0].item()*self.dt
                new_q2 = self.q1 + v_joint[1].item()*self.dt
                new_q3 = self.q1 + v_joint[2].item()*self.dt
                new_q4 = self.q1 + v_joint[3].item()*self.dt
                new_q5 = self.q1 + v_joint[4].item()*self.dt
                q_new = [new_q1, new_q2, new_q3, new_q4, new_q5]
                new_joint = [-new_q5*1000, v_joint[4]*1000, -new_q3*1000, v_joint[2]*1000, new_q4*1000, v_joint[3]*1000, 
                             new_q2*1000, v_joint[1]*1000, new_q1*1000, v_joint[0]*1000]
                # print("Joint: ", new_joint)
                # joint_queue.put(new_joint)
                # print(q_new)
                self.updateState(q_new, v_scope[3])

            except queue.Empty:
                print('No input from the joystick')
            
            joy_input.clear()

    def stop(self):
        print("Robot control stopped")
        self.running=False


if __name__ == '__main__':
    joy = JoyReader()
    robot = RobotControl()
    joy.start()
    robot.start()
    while True:
        try:
            time.sleep(0.1)
        except KeyboardInterrupt:
            break
    joy.stop()
    robot.stop()
            