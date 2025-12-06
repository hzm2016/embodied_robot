# -*- coding: utf-8 -*-
import numpy as np
import os
import sys
import time
import multiprocessing as mp
import socket
import pickle
import math
# 外部引用文件
from udp_connect import udpcomm 
from ar.src import Project_moudle

# 内部fem文件
from fem_code.fem_run import fem
from fem_code.fem_opengl_draw import draw
from udp_connect.udp_recv_steer import udp_recv_steer
from udp_connect.udp_recv_sheath_insert import udp_recv_sheath_insert

#os.chdir(os.path.split(os.path.realpath(sys.argv[0]))[0])


def rotaXmatrix(alpha):
    alpha = alpha*math.pi/180
    A=np.zeros((4,4),dtype=np.float)
    A[0][0]=1
    A[0][1]=0
    A[0][2]=0
    A[0][3]=0

    A[1][0]=0
    A[1][1]=math.cos(alpha)
    A[1][2]=-1*math.sin(alpha)
    A[1][3]=0

    A[2][0]=0
    A[2][1]=math.sin(alpha)
    A[2][2]=math.cos(alpha)
    A[2][3]=0

    A[3][0]=0
    A[3][1]=0
    A[3][2]=0
    A[3][3]=1
    return A

def rotaYmatrix(alpha):
    alpha = alpha*math.pi/180
    A=np.zeros((4,4),dtype=np.float)
    A[0][0]=math.cos(alpha)
    A[0][1]=0
    A[0][2]=math.sin(alpha)
    A[0][3]=0

    A[1][0]=0
    A[1][1]=1
    A[1][2]=0
    A[1][3]=0

    A[2][0]=-1*math.sin(alpha)
    A[2][1]=0
    A[2][2]=math.cos(alpha)
    A[2][3]=0

    A[3][0]=0
    A[3][1]=0
    A[3][2]=0
    A[3][3]=1
    return A

def rotaZmatrix(alpha):
    alpha = alpha*math.pi/180
    A=np.zeros((4,4),dtype=np.float)
    A[0][0]=math.cos(alpha)
    A[0][1]=-math.sin(alpha)
    A[0][2]=0
    A[0][3]=0

    A[1][0]=math.sin(alpha)
    A[1][1]=math.cos(alpha)
    A[1][2]=0
    A[1][3]=0

    A[2][0]=0
    A[2][1]=0
    A[2][2]=1
    A[2][3]=0

    A[3][0]=0
    A[3][1]=0
    A[3][2]=0
    A[3][3]=1
    return A
def transmatrix(dx,dy,dz):
    A=np.zeros((4,4),dtype=np.float)
    A[0][0]=1
    A[0][1]=0
    A[0][2]=0
    A[0][3]=dx

    A[1][0]=0
    A[1][1]=1
    A[1][2]=0
    A[1][3]=dy

    A[2][0]=0
    A[2][1]=0
    A[2][2]=1
    A[2][3]=dz

    A[3][0]=0
    A[3][1]=0
    A[3][2]=0
    A[3][3]=1
    return A



def udp_run(shared_udp):
    udp = udpcomm.pUDPComm()
    udp.run(shared_udp)


def udp_random(shared_udp, shared_draw):
    fr = False
    fx = False
    i = 0
    while (shared_draw['event'][0] != 1 or shared_draw['event'][1] != 1) and shared_draw['event'][0] != 5:
        if shared_draw['event'][0] == 1 and shared_draw['event'][1] == 32:
            fr = True
            fx = not fx
        if shared_draw['event'][0] == 1 and shared_draw['event'][1] == 2:
            fr = False
        if fr:
            i += 0.1
            time.sleep(0.1)
            if fx:
                x = 30 * np.cos(i)
                y = 30 * np.sin(i)
            else:
                y = 30 * np.cos(i)
                x = 30 * np.sin(i)

            shared_udp['datafloat'] = np.array([x, y, 0, 0, 1])

    shared_udp['datafloat'] = np.array([0, 0, 0, 0, 0])


def udp_recv_location(shared_location, shared_draw):
    recv_addr = ('192.168.31.48', 8080)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(recv_addr)
    s.settimeout(1)
    while (shared_draw['event'][0] != 1 or shared_draw['event'][1] != 1) and shared_draw['event'][2] != 5:
        try:
            pose__, addr = s.recvfrom(2048)
            pose_recv = pickle.loads(pose__)
            shared_location['location'] = np.array(pose_recv)  # .reshape((4, 4))
        except (Exception, BaseException) as errorstring:
            # print('%s' % str(errorstring))  # 打印错误
            pass
    s.close()


def fem1(trans_matrix, tip_matrix, mode, Is_warm_start):
    # mp.set_start_method('spawn')
    # cuda.init()
    # serialtest.SerialCom().start()
    velocity_InsertionUnit = 1  # 鞘和插入部插入速度,该值越大速度越慢，为除法,注意该值不能小于1/self.la
    velocity_sheath = 1  # 该值暂时无意义

    ctx = mp.get_context('spawn')
    manager = ctx.Manager()
    shared_fps = manager.dict({})
    shared_camera_matrix = manager.dict({})
    shared_camera_matrix['camera_matrix'] = manager.list([0, 0, 0, 1, 1, 1, 0, 1, 0])
    shared_camera_matrix['trans_matrix'] = np.eye(4, dtype=np.float32)
    shared_draw = manager.dict({})
    event = np.array([0, 0, 0])
    shared_draw['event'] = event
    shared_draw['F'] = np.zeros(100 * 6, dtype=np.float32)
    shared_draw['constraint_triangle'] = np.zeros((1000, 1), dtype=np.float32)
    shared_draw['ResultNode'] = np.zeros((100, 3), dtype=np.float32)
    shared_draw['l1'] = 99
    shared_draw['na'] = 0
    shared_draw['la'] = 1
    shared_draw['velocity_InsertionUnit'] = velocity_InsertionUnit
    shared_draw['camera_matrix'] = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0], dtype=np.float32)
    shared_draw['camera_T_start'] = np.eye(3, dtype=np.float32)
    shared_draw['set_locate'] = 0
    manager1 = ctx.Manager()
    shared_mem = manager1.dict({})
    ss = manager1.list([0])
    # ab = np.zeros(1)
    shared_fps['fps'] = manager.list([0])

    # fem(shared_mem, shared_fps, None, velocity_InsertionUnit, velocity_sheath)
    # multiprocessing.set_start_method('spawn')
    shared_udp = manager.dict({})
    shared_udp['datafloat'] = np.array([0, 0, 0, 0, 0])
    shared_udp['sheathinsert'] = tip_matrix # np.zeros((3, 4), dtype=np.float32)
    shared_location = manager1.dict({})
    # 内参， [[fx,0,cx],[0,fy,cy],[0,0,1]]
    shared_location['intrinsic'] = np.array([[320.604884143610, 0, 323.645615703525],
                                            [0, 321.838038649933, 238.098053365179],
                                            [0, 0, 1]], dtype=np.float32)
    # 定位和欧拉转角，x,y,z,rx,ry,rz
    # shared_location['location'] = np.array([[-0.999059796,-0.042072266,0.010458697,-0.916783988],[0.003219745,0.168573037,0.985683858,9.344343185],[-0.043233007,0.984790862,-0.168279082,-1.924159646],[0,0,0,1]], dtype=np.float32)
    shared_location['location'] = 10 * np.array([-0.916783988, 9.344343185, -1.924159646, -1.739889264, 0.010458888, 3.099505663]) # 单位改成mm

    process = [ctx.Process(target=fem, args=(
                shared_mem, shared_fps, None, velocity_InsertionUnit, velocity_sheath, shared_camera_matrix, trans_matrix,
                tip_matrix, shared_draw, shared_udp, mode, shared_location, Is_warm_start)),  # 有限元主函数
               ctx.Process(target=draw, args=(trans_matrix, shared_draw, mode, shared_location)),  # 绘制主函数
               # ctx.Process(target=Project_moudle.AR_show, args=(shared_camera_matrix,)),  # 输出到AR
               # ctx.Process(target=udp_recv_location, args=(shared_location, shared_draw,)),  # 读取定位
               # ctx.Process(target=udp_run, args=(shared_udp,)),  # 改用新的和主手的通信函数udp_recv_steers
               ctx.Process(target=udp_recv_steer, args=(shared_udp, shared_draw,)),  # 主手通信 # 注意该程序数据排列和原版不同
               ctx.Process(target=udp_recv_sheath_insert, args=(shared_udp, shared_draw,)),  # only sheath insert
               ]

    [p.start() for p in process]
    [p.join() for p in process]


if __name__ == '__main__':
    path_root = os.getcwd()
    Is_warm_start = False
    if len(sys.argv) > 1:
        Is_warm_start = sys.argv[1]
    trans_matrix = np.load("/home/p9/Projects/embodiedrobot/navigation/data/trans_matrix.npy")
    tip_matrix = np.load("/home/p9/Projects/embodiedrobot/navigation/data/tip2cam.npy")
    points = np.load("/home/p9/Projects/embodiedrobot/navigation/data/transed_points.npy")

    #tip_matrix = [[ 0.39885423,0.4178009,  0.81630736,-0.08214231],[ 0.70114554, -0.71267332, 0.02217387, -0.03635358 ],[0.59102474,  0.56350612, -0.577192,  0.55093282],[ 0.   ,       0.       ,   0.        ,  1.        ]]
    tip_matrix = tip_matrix@transmatrix(0.05,-0.037,0.032)
    tip_matrix = tip_matrix@rotaYmatrix(45)@rotaZmatrix(15)
    # tip_matrix = tip_matrix@
    #trans_matrix = np.eye(4, dtype=np.float32)
    #tip_matrix = None

    mode = 1
    fem1(trans_matrix, tip_matrix, mode, Is_warm_start)
