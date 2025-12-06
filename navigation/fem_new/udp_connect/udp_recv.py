import socket

if __name__ == '__main__':
    recv_addr = ('192.168.3.130', 8080)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    send_addr = ('192.168.3.231', 8080)
    s.bind(send_addr)
    while 1:
        data, addr = s.recvfrom(2048)
        s.sendto(data, recv_addr)
        if not data:
            break
        print(data.decode())
    s.close()
