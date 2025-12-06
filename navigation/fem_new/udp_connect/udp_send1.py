import socket
import time
import numpy as np
import pickle

if __name__ == '__main__':
    recv_addr = ('', 8080)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    send_addr = ('127.0.0.1', 8001)
    s.bind(recv_addr)
    pose = np.array([0, 1, 2, 3, 4, 5, 6])
    while 1:
        # a = input('请输入：')  # 为了每次在此停顿一下
        data = time.time()
        pose_ = pickle.dumps(pose)
        s.sendto(pose_, send_addr)
        pose__, addr = s.recvfrom(2048)
        pose_send = pickle.loads(pose__)
        if not a:
            break
        print(pose_send)
    s.close()
