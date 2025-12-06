import socket
import time
import numpy as np
import robot_new_Jacob
import cv2
from joyread import JoyReader

class RobotConnect:
    def __init__(self) -> None:
        self.robosock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
    def connect(self, host='192.168.3.136', port=8000):
        self.robosock.connect((host, port))
        print(f'Connected to {host}:{port}')
        
    def reset(self):
        message = 'RE'
        self.robosock.sendall(message.encode('utf-8'))
        
    def close(self):
        self.robosock.close()
        print('Connection closed.')
        
    def move_motor(self,m1,m2,m3,m4,m5):
        m1 = -1*m1/55*47000
        m2 = (55-m2)/55*47000
        m3 = m3/55*47000
        m4 = (m4-55)/55*47000
        m5 = (m5-55)/55*47000

        m1 = round(m1)
        m2 = round(m2)
        m3 = round(m3)
        m4 = round(m4)
        m5 = round(m5)

        if m1>47000:
            m1 = 47000
        if m1<-47000:
            m1 = -47000
        if m2>47000:
            m2 = 47000
        if m2<-47000:
            m2 = -47000
        if m3>47000:
            m3 = 47000
        if m3<-47000:
            m3 = -47000
        if m4>47000:
            m4 = 47000
        if m4<-47000:
            m4 = -47000
        message = f'{m1},{m2},{m3},{m4},{m5}'
        print(f'Sent: {message}')
        self.robosock.sendall(message.encode('utf-8'))
        return message

if __name__ == '__main__':
    joy_reader = JoyReader()
    joy_reader.start()

    robot = RobotConnect()
    robot.connect()
    # robot.reset()
    # print('reset')
    # time.sleep(5)
    init_pos = [20/1000,55/2/1000,55/2/1000,55/2/1000,55/2/1000]
    # init_pos = [0,0,0,0,0]
    message = robot.move_motor(init_pos[0],init_pos[1],init_pos[2],init_pos[3],init_pos[4])
    print('move')
    time.sleep(3)
    
    # X = [27.50/1000, 27.50/1000, 27.50/1000, 27.50/1000]
    v_s = np.matrix([0., 0.,0.000, 0.0000]).T
    gama_init = 0.5
    cap = cv2.VideoCapture(2)

    forward_cnt = 0

    while True: 
        ret, src = cap.read()
        # src = src[830:1100]
        # src = cv2.imread("../3.jpg") 
        src = cv2.resize(src, (1280, 720))   
        
        v_e = robot_new_Jacob.GetRCM_scp2end(gama_init, v_s) # 获得持镜机构末端速度
        print("ve_",v_e)
        v_joint = robot_new_Jacob.GetEnd2JointVel(v_e, init_pos[1:]) # 获得各关节移动速度
        print(v_joint)
        print(robot_new_Jacob.GetEndVel(v_joint , init_pos[1:]))
        #  = [X[0]+v_joint[0].item(),X[1]+v_joint[1].item(),X[2]+v_joint[2].item(),X[3]+v_joint[3].item()]

        new_pose = [0,init_pos[1]+v_joint[2].item(),init_pos[2]+v_joint[3].item(),init_pos[3]+v_joint[0].item(),init_pos[4]+v_joint[1].item()]
        init_pos = new_pose

        print(new_pose)
        message = robot.move_motor(new_pose[0]*1000,new_pose[1]*1000,new_pose[2]*1000,new_pose[3]*1000,new_pose[4]*1000)

        v_s = np.matrix([0., 0.,0.000, 0.000]).T
        
        # info="F:{},B:{},L/R:{},U/D:{}".format()
        info = "{},{},{},{},{}".format(new_pose[0],new_pose[1],new_pose[2],new_pose[3],new_pose[4])
        
        cv2.putText(src, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Image", src)
        key = cv2.waitKey(1)
        if key == 27:
            break
        elif key == ord('w'):
            v_s = np.matrix([0., 0.,0.000, 0.00001]).T
        elif key == ord('s'):
            v_s = np.matrix([0., 0.,0.000, -0.00001]).T
        elif key == ord('a'):
            v_s = np.matrix([0., 0.,0.0001, 0.000]).T
        elif key == ord('d'):
            v_s = np.matrix([0., 0.,-0.0001, 0.000]).T
        else:
            v_s = np.matrix([0., 0.,0.000, 0.000]).T
        

    robot.close()
    pass


