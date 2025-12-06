import socket
import time

robosock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
robosock.connect(('192.168.1.136',8000))
print('Connected to 192.168.1.136:8000')

message = '3'
robosock.sendall(message.encode('utf-8'))

time.sleep(30)

print('reset complete!')
message = 'q'
robosock.sendall(message.encode('utf-8'))
time.sleep(1)
message = '1'
robosock.sendall(message.encode('utf-8'))
time.sleep(1)

robosock.close()