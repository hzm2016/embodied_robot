import numpy as np
import os
import sys
import time
import multiprocessing as mp
import socket
import pickle
# import keyboard


def udp_recv_sheath_insert(shared_udp, shared_draw):
    recv_addr = ('127.0.0.1', 9527)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(1)
    s.bind(recv_addr)
    while (shared_draw['event'][0] != 1 or shared_draw['event'][1] != 1) and shared_draw['event'][2] != 5:
        try:
            input_recv, addr = s.recvfrom(1024)
            # pose_recv = pickle.loads(pose__)
            # shared_location['location'] = np.array(pose_recv)  # .reshape((4, 4))
            recv_str = ''.join(input_recv.decode('utf-8').split())
            arr = np.array(recv_str.rstrip('\n').split(','), dtype=np.float32).reshape(3, 4)
            shared_udp['sheathinsert'] = arr
            # hDev.hPosition[0], hDev.hPosition[2], sim_sheathinsert, steer_mode
            print(arr)
        except(Exception, BaseException) as errorstring:
            print('%s' % str(errorstring))  # 打印错误
            pass
    s.close()


if __name__ == '__main__':
    shared_udp = {'datafloat': [0,0,0,0]}
    shared_draw = {'datafloat': [0, 0, 0, 0]}
    udp_recv_sheath_insert(shared_udp, shared_draw)
