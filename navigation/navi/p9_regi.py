# import keyboard
import os
from statistics import mode
import sys
import numpy as np
import time
# from registration import Registration
# import rgbdcamera as rgbdcam
from realman_command import RMCommand
from realman_kinematics import forward_kinematics as fk

BASE_PATH = "/home/mingcong/projects/p9/regesiteration/data/"

def pose_to_transformation_matrix(pose):
    """
    Convert pose [x,y,z,rx,ry,rz] to 4x4 transformation matrix
    Args:
        pose: list or numpy array [x,y,z,rx,ry,rz] 
            where rx,ry,rz are rotation angles in radians
            x,y,z are in millimeters
    Returns:
        4x4 homogeneous transformation matrix
    """
    x, y, z = pose[0:3]
    rx, ry, rz = pose[3:6]

    # Calculate rotation matrices for each axis
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])
    
    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    
    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ])

    # Combined rotation matrix
    R = Rz @ Ry @ Rx

    # Create 4x4 transformation matrix
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = [x, y, z]
    
    return T

if __name__ == '__main__':
    rm_command = RMCommand()
    rm_command.connect_tcp_socket()
    
    # [-2.120208522030192, -0.7157071663653146, 2.308128122592421, -0.3613006084553462, -0.6607816548050531, 1.6056156453721835, -0.6343573699298589]
    # rgbdcamera = rgbdcam.RGBDCamera()	
    # register  = Registration(rgbdcamera)

    # trans_matrix, transed_points = register.main_registration()
    trans_matrix = np.load("trans_matrix.npy")
    print(trans_matrix)

    np.save(os.path.join(BASE_PATH, 'trans_matrix_head2camera.npy'),trans_matrix)

    model2camera = np.load("/home/mingcong/projects/p9/regesiteration/data/trans_matrix_head2camera.npy")
    model2camera[:3,3] = model2camera[:3,3]*100
    camera2model = np.linalg.inv(model2camera) # 可变?
    ee2camera = pose_to_transformation_matrix([50,-60,-30,np.pi/2,0,np.pi/2]) # 不可变，以远离手术台和相机的底角为坐标轴
    camera2ee = np.linalg.inv(ee2camera)
    ee2model = camera2model @ ee2camera

    joint = rm_command.read_j()
    print(joint)
    
    joint_rad = [j*np.pi/180 for j in joint]
    ee2base = fk(joint_rad) # 可变
    camera2base = ee2base @ camera2ee
    model2base = camera2base @ model2camera
    base2model = np.linalg.inv(model2base)

    np.save(os.path.join(BASE_PATH, 'trans_matrix_base2model.npy'),base2model)
    np.save(os.path.join(BASE_PATH, 'trans_matrix_ee2model.npy'),ee2model)
    print("finish")
    # del register
    # del rgbdcamera
    # print(transed_points)
