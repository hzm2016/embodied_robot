"""
主程序入口
实时姿态估计与碰撞检测系统
"""
import os
import sys
from typing import Any, Dict
from Robotic_Arm.rm_robot_interface import *
import numpy as np
from math import pi
from RCM.Kinematics import GetEnd2JointVel
import time


def main():  
    Realman = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    handle = Realman.rm_create_robot_arm("192.168.2.18", 8080, 3)
    arm_data = Realman.rm_get_arm_current_trajectory()
    print("now_degrees:",arm_data["data"])  
    # theta = np.deg2rad(arm_data["data"])   

    theta = np.array([0, pi/4, 0, pi/2, 0, -pi/4, -pi/6])     
    res = Realman.rm_movej(np.rad2deg(theta),10,0,0,1)    
    # print("movej result:",res)    
    time.sleep(5)
    
    for i in range(50):  
        print(i, theta)   
        vend = np.array([0.0, 0.0, 0.1, 0, 0, 0])  # 末端速度，单位m/s，rad/s
        vjoint = GetEnd2JointVel(vend, theta)    
        theta = theta + vjoint.flatten()*0.1  # 关节速度，单位rad/s  
        # Realman.rm_movej_canfd(np.rad2deg(theta),False,0,1,50)
        res = Realman.rm_movej(np.rad2deg(theta),10,0,0,1)
        print("movej result:",res)
    
    # for i in range(1000):
    #     print(theta)
    #     vend = np.array([0.01, 0.0, 0, 0, 0, 0])  # 末端速度，单位m/s，rad/s
    #     vjoint = GetEnd2JointVel(vend, theta)
    #     theta = theta + vjoint.flatten()*0.01  # 关节速度，单位rad/s
    #     Realman.rm_movej_canfd(np.rad2deg(theta),False,0,1,50)

        Realman.rm_destroy()  

if __name__ == "__main__":
    main()