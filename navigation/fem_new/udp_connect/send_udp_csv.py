import socket
import time
import numpy as np
import pickle
import pandas as pd
import keyboard

if __name__ == '__main__':
    POSE_DIR = 'D:/陈浩/2022年研究/FEM_Bronch/20221104管线碰撞-肺部支气管/data/predict_pose.csv'
    # recv_addr = ('', 8080)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    send_addr = ('192.168.3.231', 8080)
    # s.bind(recv_addr)
    pose_list = np.array(pd.read_csv(POSE_DIR, sep=',', header=None))
    while True:
        for i in range(3865):
            pose = pose_list[i, :]
            # a = input('')
            # data = time.time()
            time.sleep(0.02)
            pose_ = pickle.dumps(pose)
            s.sendto(pose_, send_addr)
            # pose__, addr = s.recvfrom(2048)
            # pose_send = pickle.loads(pose__)
            if keyboard.is_pressed('esc'):
                break
            # print(pose__)
        if keyboard.is_pressed('esc'):
            break
    s.close()
