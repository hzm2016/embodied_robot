import socket
import time
import numpy as np
import cv2
from joyread import JoyReader
from robot_class import RobotControl
import matplotlib.pyplot as plt

class RobotConnect:
    def __init__(self) -> None:
        self.robosock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
    def connect(self, host='192.168.1.136', port=8000):
        self.robosock.connect((host, port))
        print(f'Connected to {host}:{port}')
        
    def reset(self):
        message = '3'
        self.robosock.sendall(message.encode('utf-8'))

    def mode_CSP(self):
        message = 'q'
        self.robosock.sendall(message.encode('utf-8'))
        time.sleep(1)
        message = '1'
        self.robosock.sendall(message.encode('utf-8'))
        
    def close(self):
        self.robosock.close()
        print('Connection closed.')
        
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
            
        if m4<-17:
            m4 = -17
            
        if m5>25:
            m5 = 25
            
        if m5<-17:
            m5 = -17
            
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
        return message
    
def checkMotor(joint_pos):
    out = False
    if -joint_pos[0] > 0 or -joint_pos[0] < -50/1000:
        out = True
        print('motor 1 out of range')
    elif -joint_pos[1] > 22/1000 or -joint_pos[1] < -22/1000:
        out = True
        print('motor 2 out of range')
    elif joint_pos[2] > 22/1000 or joint_pos[2] < -22/1000:
        out = True
        print('motor 3 out of range')
    elif joint_pos[3] > 25/1000 or joint_pos[3] < -17/1000:
        out = True
        print('motor 4 out of range')
    elif joint_pos[4] > 25/1000 or joint_pos[4] < -17/1000:
        out = True
        print('motor 5 out of range')

    return out

if __name__ == '__main__':
    dt = 0.01
    joy_reader = JoyReader()
    joy_reader.start()
    
    soc = RobotConnect()
    soc.connect()
    print('reseting')
    soc.reset()
    time.sleep(35)

    soc.mode_CSP()
    
    # X = [27.50/1000, 27.50/1000, 27.50/1000, 27.50/1000]
    v_s = [0., 0.,0.000, 0.0000]
    gama_init = 0.09/0.1945
    cap = cv2.VideoCapture(1)
    joint_pos = [0,0,0,0,0] # m1:j5, m2:j3, m3:j4, m4:j2, m5:j1
    forward_cnt = 0
    robot = RobotControl(q_init=joint_pos, gamma=gama_init)
    traj_vel = []
    traj_q = []
    
    while True: 
        t0 = time.time()
        # ret, src = cap.read()
        # src = src[830:1100]
        src = cv2.imread("3.jpg") 
        src = cv2.resize(src, (1280, 720))
        
        v_joint = robot.getJointVelocity(v_s) # 获得各关节移动速度

        # TODO: The readings of joint position and joint velocity should be obtained from the motor
        new_pose = [joint_pos[4]+v_joint[0].item()*dt,joint_pos[3]+v_joint[1].item()*dt,joint_pos[1]+v_joint[2].item()*dt,
                    joint_pos[2]+v_joint[3].item()*dt,joint_pos[0]+v_joint[4].item()*dt]
        temp = joint_pos[:]
        joint_pos = [new_pose[4], new_pose[2], new_pose[3], new_pose[1], new_pose[0]]
        
        robot.updateState(new_pose, v_s[3])
        if not checkMotor(joint_pos):
            message = soc.move_motor(-joint_pos[0]*1000, v_joint[4].item()*1000, -joint_pos[1]*1000, -v_joint[2].item()*1000, joint_pos[2]*1000, v_joint[3].item()*1000,
                                    joint_pos[3]*1000, -v_joint[1].item()*1000, joint_pos[4]*1000, -v_joint[0].item()*1000)
        else:
            v_s=[0, 0, 0, 0]
            new_pose = [temp[4], temp[3], temp[1], temp[2], temp[0]]
            joint_pos = [new_pose[4], new_pose[2], new_pose[3], new_pose[1], new_pose[0]]
        robot.updateState(new_pose, v_s[3])
        # TODO: Send T05 to the navigation program

        v_s = [0., 0.,0.000, 0.000]
        
        # info="F:{},B:{},L/R:{},U/D:{}".format()
        info = "{},{},{},{},{}".format(new_pose[0],new_pose[1],new_pose[2],new_pose[3],new_pose[4])
        
        cv2.putText(src, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Image", src)
        key = cv2.waitKey(1)
        if key == 27:
            break
        elif key == ord('w'):
            v_s = [0., 0., 0., 0.025]
        elif key == ord('s'):
            v_s = [0., 0., 0., -0.025]
        elif key == ord('a'):
            v_s = [0., 0.05, 0., 0.]
        elif key == ord('d'):
            v_s = [0., -0.05, 0., 0.]
        elif key == ord('i'):
            v_s = [0.06, 0., 0., 0.]
        elif key == ord('k'):
            v_s = [-0.06, 0., 0., 0.]
        
        else:
            v_s = [0., 0., 0., 0.]
        # time.sleep(0.01)
        delta_t=time.time()-t0


    soc.close()
    