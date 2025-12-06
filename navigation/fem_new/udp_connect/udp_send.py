import socket
import time

if __name__ == '__main__':
    recv_addr = ('10.0.10.91', 8080)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    send_addr = ('10.11.10.135', 8080)
    s.bind(recv_addr)
    while 1:
        a = input('请输入：')  # 为了每次在此停顿一下
        data = time.time()
        if not a:
            data = a
        s.sendto(str(data).encode("utf-8"), send_addr)
        t, addr = s.recvfrom(2048)
        t1 = time.time()
        if not a:
            break
        print(t)
        print((t1-float(t))/2.0)
    s.close()
