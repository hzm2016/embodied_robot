import socket
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import threading
# import queue
from queue import Queue, Empty
import serial
from math import sin, cos, tan, atan, pi
import redis

joy_input = threading.Event()
reset_event = threading.Event()
forward_event = threading.Event()
velocity_queue = Queue()
joint_queue = Queue()

class JoyReader(threading.Thread):
    def __init__(self, port='/dev/ttyACM0', baudrate=9600, timeout=1):
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
            
            print('Joyreader Thread started!')
            while self.running:
                if reset_event.is_set():
                    reset_event.wait()
                    print('Joyreader Thread waitint for reset to complete...')
                    time.sleep(5)
                    print('Joyreader Thread restart!')
                    reset_event.clear()
                elif forward_event.is_set():
                    forward_event.wait()
                    time.sleep(5)
                    print('Joyreader Thread restart!')
                    forward_event.clear()
                try:
                    data = self.read_serial_data()
                    if data[0] == 0:
                        print('moving forward')
                        v_scope = [0, 0, 0, 0.1]
                        
                        velocity_queue.put(v_scope)
                    elif data[1] == 0:
                        print('moving backward')
                        v_scope = [0, 0, 0, -0.1]
                        velocity_queue.put(v_scope)
                    elif data[2] > 0 and abs(data[2]) > abs(data[3]):
                        print('turning left')
                        v_scope = [0, 0.2, 0, 0]
                        velocity_queue.put(v_scope)
                    elif data[2] < 0 and abs(data[2]) > abs(data[3]):
                        print('turning right')
                        v_scope = [0., -0.2, 0., 0.]
                        velocity_queue.put(v_scope)
                    elif data[3] > 0 and abs(data[3]) > abs(data[2]):
                        print('moving upward')
                        v_scope = [-0.2, 0., 0., 0.]
                        velocity_queue.put(v_scope)
                    elif data[3] < 0 and abs(data[3]) > abs(data[2]):
                        print('moving downward')
                        v_scope = [0.2, 0., 0., 0.]
                        velocity_queue.put(v_scope)
                    # else:
                    #     v_scope = [0., 0., 0., 0.]
                        # velocity_queue.put(v_scope)
                    # print queue size
                    # print("!!!!!!!!!!!!!!!!!!!!!!")
                    # print(f'Queue size: {velocity_queue.qsize()}')
                    joy_input.set()
                
                except TypeError:
                    print("No input from joystick")
                
    def stop(self):
        self.running = False
        if self.ser:
            print('Joystick stopped.')
            self.ser.close()


class RobotConnect(threading.Thread):
    def __init__(self):
        super().__init__()
        self.robosock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.running = True

    def connect(self, host='192.168.1.136', port=8000):
        self.robosock.connect((host, port))
        print(f'Connected to {host}:{port}')
        
    def reset(self):
        self.move_motor(0,3,0,3,0,3,0,3,0,3)

    def move_Forward(self, d_in):
        self.move_motor(-d_in,3,0,0,0,0,0,0,0,0)
    
    def mode_CSP(self):
        message = 'q'
        self.robosock.sendall(message.encode('utf-8'))
        time.sleep(1)
        message = '1'
        self.robosock.sendall(message.encode('utf-8'))
        time.sleep(0.1)
        
    def close(self):
        self.robosock.close()
        print('Connection closed.')
        self.running = False

    def run(self):
        self.connect()
        # self.reset()
        # time.sleep(35)
        # self.mode_CSP()

        while self.running:
            try:
                joints=joint_queue.get()
                self.move_motor(joints[0],joints[1],joints[2],joints[3],joints[4],joints[5],
                                joints[6],joints[7],joints[8],joints[9])
            except Empty:
                print('no data from robot control')

    def move_motor(self,m1,v1,m2,v2,m3,v3,m4,v4,m5,v5):
        if m1>0:
            m1 = 0

        if m1<-50:
            m1 = -50
            
        if m2>22:
            m2 = 22
            
        if m2<-22:
            m2 = -22
            
        if m3>22:
            m3 = 22
            
        if m3<-22:
            m3 = -22
            
        if m4>24:
            m4 = 25
            
        if m4<-24:
            m4 = -25
            
        if m5>25:
            m5 = 25
            
        if m5<-25:
            m5 = -25
            
        m1 = m1*3072*20/64
        m2 = m2*3072*20/64
        m3 = m3*3072*20/64
        m4 = m4*3072*20/64
        m5 = m5*3072*20/64

        m1 = round(m1)
        m2 = round(m2)
        m3 = round(m3)
        m4 = round(m4)
        m5 = round(m5)
        v1 = round(abs(v1*3072*20/64000), 4)
        v2 = round(abs(v2*3072*20/64000), 4)
        v3 = round(abs(v3*3072*20/64000), 4)
        v4 = round(abs(v4*3072*20/64000), 4)
        v5 = round(abs(v5*3072*20/64000), 4)

        message = f'{m1},{v1},{m2},{v2},{m3},{v3},{m4},{v4},{m5},{v5}'
        print(f'Sent: {message}')
        self.robosock.sendall(message.encode('utf-8'))

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
        self.yeta1 = atan((self.q2-self.q1)/0.055)
        self.yeta2 = atan((self.q3-self.q4)/0.055)
        self.q_pos = [self.q1, self.q2, self.q3, self.q4, self.q5]
        self.d_in = gamma*0.1945
        self.T5rcm=np.matrix([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., -self.d_in], [0., 0., 0., 1.]])
        self.test_mode=testmode
        self.T04 = None
        self.T05 = None
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
    
    
    def updateState(self, X):
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
        self.yeta1 = atan((self.q2-self.q1)/0.055)
        self.yeta2 = atan((self.q3-self.q4)/0.055)
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
        self.T04 = T01 * T12 * T23 * T34
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
            joy_input.clear()
            
            try:
                v_scope = velocity_queue.get()
                print("-----------v---------------")
                # print queue size
                print(f'Queue size: {velocity_queue.qsize()}')
                v_joint = self.getJointVelocity(v_scope)
                new_q1 = self.q1 + v_joint[0].item()*self.dt
                new_q2 = self.q2 + v_joint[1].item()*self.dt
                new_q3 = self.q3 + v_joint[2].item()*self.dt
                new_q4 = self.q4 + v_joint[3].item()*self.dt
                new_q5 = self.q5 + v_joint[4].item()*self.dt

                q_new = [new_q1, new_q2, new_q3, new_q4, new_q5]

                new_joint = [-new_q5*1000, v_joint[4].item()*1000, -new_q3*1000, v_joint[2].item()*1000, new_q4*1000, v_joint[3].item()*1000, 
                             new_q2*1000, v_joint[1].item()*1000, new_q1*1000, v_joint[0].item()*1000]
                if not checkMotor(new_joint):
                    joint_queue.put(new_joint)
                else:
                    new_q1 = self.q1
                    new_q2 = self.q2
                    new_q3 = self.q3
                    new_q4 = self.q4
                    new_q5 = self.q5
                    q_new = [new_q1, new_q2, new_q3, new_q4, new_q5]
                    new_joint = [-new_q5*1000, 0, -new_q3*1000, 0, new_q4*1000, 0, new_q2*1000, 0, new_q1*1000, 0]
                    joint_queue.put(new_joint)
                self.updateState(q_new)

            except Empty:
                print('No input from the joystick')

    def stop(self):
        print("Robot control stopped")
        self.running=False

class Publisher:
    def __init__(self, redis_host='localhost', redis_port=6379, redis_db=0, key="endo_position"):
        # Initialize Redis connection and configuration
        self.redis_client = redis.StrictRedis(host=redis_host, port=redis_port, db=redis_db)
        self.key = key
        self.endo_start = [0, 0, 200]  # Default starting position
        self.endo_end = [100, 0, 200]  # Default ending position
        self.lock = threading.Lock()  # Lock for thread-safe updates to positions

    def update_positions(self, start, end):
        # Update the endo_start and endo_end positions safely
        with self.lock:
            self.endo_start = start
            self.endo_end = end

    def publish_messages(self):
        while True:
            # Safely read positions
            with self.lock:
                start = self.endo_start
                end = self.endo_end

            # Prepare the value to publish
            value = f"{start[0]},{start[1]},{start[2]},{end[0]},{end[1]},{end[2]}"
            self.redis_client.set(self.key, value)
            # print(f"Data sent to Redis: [{self.key}] = [{value}]")


def checkMotor(joint_pos):
    out = False
    if joint_pos[0] > 0 or joint_pos[0] < -50:
        out = True
    elif joint_pos[2] > 22 or joint_pos[2] < -22:
        out = True
    elif joint_pos[4] > 22 or joint_pos[4] < -22:
        out = True
    elif joint_pos[6] > 22 or joint_pos[6] < -15:
        out = True
    elif joint_pos[8] > 22 or joint_pos[8] < -15:
        out = True

    return out

if __name__ == '__main__':
    gamma = 0.00/0.1945
    d_in = 20
    joy_reader = JoyReader()
    joy_reader.start()
    soc = RobotConnect()
    soc.start()
    robot = RobotControl(gamma=gamma)
    robot.start()

    # Create a Publisher instance
    publisher = Publisher()
    # Create a thread for publishing messages
    publisher_thread = threading.Thread(target=publisher.publish_messages)
    # Start the publisher thread
    publisher_thread.start()

    cap = cv2.VideoCapture(2)
    while True:
        endo_start = robot.T04[0:3,[3]].T.tolist()[0]
        endo_end = robot.T05[0:3,[3]].T.tolist()[0]
        angle_ver = robot.yeta2
        angle_hor = robot.yeta1
        angle_ver = round(angle_ver*180/pi,4)
        angle_hor = round(angle_hor*180/pi,4)
        publisher.update_positions(endo_start, endo_end)
        ret, src = cap.read()
        # src = cv2.imread("3.jpg")
        src = cv2.resize(src, (1280, 720))
        cv2.putText(src, f"Yaw: {angle_hor} deg", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
        cv2.putText(src, f"Pitch: {angle_ver} deg", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
        cv2.circle(src, (640, 360), 3, (0, 0, 255), -1)
        cv2.imshow("Image", src)
        

        key = cv2.waitKey(1)
        if key == 27:
            break
        elif key == ord('q'):
            print('reseting')
            soc.mode_CSP()
            soc.reset()
            robot.updateState([0,0,0,0,0])
            reset_event.set()
            
        elif key == ord('f'):
            d_in = 20
            soc.move_Forward(d_in)
            robot.updateState([0,0,0,0,d_in/1000])
            # time.sleep(5)
            forward_event.set()
    
    cv2.destroyAllWindows()
    joy_reader.stop()
    soc.close()
    robot.stop()