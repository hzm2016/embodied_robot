# -*- coding: utf-8 -*-

import socket  #导入socket模块
import time #导入time模块
      #server 接收端
      # 设置服务器默认端口号
import numpy as np
import time
import threading

lock = threading.Lock()
hGimbalAngle = [0,0]
steermode = 0
sim_sheathinsert = 0
sim_tubeinsert = 0


class pUDPComm():
    def __init__(self):
        PORT = 7890
            # 创建一个套接字socket对象，用于进行通讯
            # socket.AF_INET 指明使用INET地址集，进行网间通讯
            # socket.SOCK_DGRAM 指明使用数据协议，即使用传输层的udp协议
        self.connect = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        address = ("127.0.0.1", PORT)  
        self.connect.bind(address)  # 为服务器绑定一个固定的地址，ip和端口
        self.connect.settimeout(10)  #设置一个时间提示，如果10秒钟没接到数据进行提示

    def run(self, shared_udp):
        while True:
            #正常情况下接收数据并且显示，如果10秒钟没有接收数据进行提示（打印 "time out"）
            #当然可以不要这个提示，那样的话把"try:" 以及 "except"后的语句删掉就可以了
            try:  
                now = time.time()  #获取当前时间

                                # 接收客户端传来的数据 recvfrom接收客户端的数据，默认是阻塞的，直到有客户端传来数据
                                # recvfrom 参数的意义，表示最大能接收多少数据，单位是字节
                                # recvfrom返回值说明
                                # receive_data表示接受到的传来的数据,是bytes类型
                                # client  表示传来数据的客户端的身份信息，客户端的ip和端口，元组
                data, client = self.connect.recvfrom(1024)
                #print(data)
                #print(len(data))
                if len(data)==50:
                    data = data.decode().strip().split(',')
                    datafloat = []
                    for n in data:
                        datafloat.append(float(n.replace(" ","")))

                    shared_udp['datafloat'] = datafloat
                    #print(datafloat)
                    #print(sim_sheathinsert,sim_tubeinsert,hGimbalAngle,steermode)
                else:
                    print("Length Error:",len(data))

            except socket.timeout:  #如果10秒钟没有接收数据进行提示（打印 "time out"）
                print ("time out")


class UDPComm(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        PORT = 7890
            # 创建一个套接字socket对象，用于进行通讯
            # socket.AF_INET 指明使用INET地址集，进行网间通讯
            # socket.SOCK_DGRAM 指明使用数据协议，即使用传输层的udp协议
        self.connect = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        address = ("127.0.0.1", PORT)  
        self.connect.bind(address)  # 为服务器绑定一个固定的地址，ip和端口
        self.connect.settimeout(10)  #设置一个时间提示，如果10秒钟没接到数据进行提示

    def run(self):
        while True:
            #正常情况下接收数据并且显示，如果10秒钟没有接收数据进行提示（打印 "time out"）
            #当然可以不要这个提示，那样的话把"try:" 以及 "except"后的语句删掉就可以了
            try:  
                now = time.time()  #获取当前时间

                                # 接收客户端传来的数据 recvfrom接收客户端的数据，默认是阻塞的，直到有客户端传来数据
                                # recvfrom 参数的意义，表示最大能接收多少数据，单位是字节
                                # recvfrom返回值说明
                                # receive_data表示接受到的传来的数据,是bytes类型
                                # client  表示传来数据的客户端的身份信息，客户端的ip和端口，元组
                data, client = self.connect.recvfrom(1024)
                print(data)
                if len(data)==59:
                    data = data.decode().strip().split(',')
                    datafloat = []
                    for n in data:
                        datafloat.append(float(n.replace(" ","")))

                    hGimbalAngle[0] =datafloat[0]   # 104 - 158
                    hGimbalAngle[1] =datafloat[1]   # -79 - -132

                    steermode= int(datafloat[4])
                    
                    sim_sheathinsert = datafloat[2]
                    sim_tubeinsert = datafloat[3]

                    #print(sim_sheathinsert,sim_tubeinsert,hGimbalAngle,steermode)

            except socket.timeout:  #如果10秒钟没有接收数据进行提示（打印 "time out"）
                print ("time out")

if __name__ == "__main__":
    UDPComm().start()