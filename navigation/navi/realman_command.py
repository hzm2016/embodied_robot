import socket
import json
import numpy as np
import sys
from realman_kinematics import forward_kinematics as fk
import time
class RMCommand:
    def __init__(self):
        self.rlm_port = 8080
        self.rlm_ip = "192.168.2.18"
        self.rlm_socket = None
        self.command_msg = {}
        self.return_msg = {}

    def connect_tcp_socket(self):
        try:
            self.rlm_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.rlm_socket.connect((self.rlm_ip, self.rlm_port))
            print("Robot connected!")
        except Exception as e:
            print(f"ERROR! Can't connect robot: {e}")
            sys.exit(0)

    def set_high_speed_eth(self):
        # Open high speed ethernet
        self.command_msg = {
            "command": "set_high_speed_eth",
            "mode": 1
        }
        self._send_and_receive()

        # Set IP
        self.command_msg = {
            "command": "set_high_ethernet",
            "ip": "192.168.1.18",
            "mask": "255.255.255.0",
            "gateway": "192.168.1.1"
        }
        self._send_and_receive()

        # Save info
        self.command_msg = {
            "command": "save_device_info_all"
        }
        self._send_and_receive()
        
        print("Successfully set high speed ethernet! Change port and restart the robot.")

    def read_j(self):
        self.command_msg = {
            "command": "get_joint_degree"
        }
        self._send_and_receive()
        
        if not self.return_msg.get("joint"):
            return
            
        cmd_joints = np.array(self.return_msg["joint"])
        # print(f"ReadJ OK!\t{cmd_joints}")
        return (cmd_joints/1000)
    
    def read_arm_state(self):
        self.command_msg = {
            "command": "get_current_arm_state"
        }
        self._send_and_receive()
        
        if not self.return_msg.get("arm_state"):
            return
            
        cmd_joints = self.return_msg["arm_state"]
        joint = cmd_joints['joint']
        joint = [j/1000 for j in joint]
        # get the pose
        pose = cmd_joints['pose']
        pose = [p/1000000 if i < 3 else p/1000 for i, p in enumerate(pose)]
        
        return pose,joint
    
    def move_j(self, joints, velo):

        self.command_msg = {"command":"movej","joint":[j*1000 for j in joints],"v":velo,"r":0,"trajectory_connect":0}
        print(self.command_msg)
        response = self._send_and_receive()
        print(response)
        if response.get("receive_state"):
            print(f"MoveJ OK!\t{response}")
        else:
            print(f"ERROR! MoveJ False!\t{response}")
            sys.exit(0)

    def move_l(self, pose, velo):
        self.command_msg = {
            "command": "movel",
            "pose": [int(1000000 * p) if i < 3 else int(1000 * p) for i, p in enumerate(pose)],
            "v": velo,
            "r": 0
        }
        response = self._send_and_receive()
        
        if response and response.get("trajectory_state"):
            print(f"MoveL OK!\t{response}")
        else:
            print(f"ERROR! MoveL False!\t{response}")
            sys.exit(0)

    def move_jp(self, pose, velo):
        self.command_msg = {
            "command": "movej_p",
            "pose": [int(1000000 * p) if i < 3 else int(1000 * p) for i, p in enumerate(pose)],
            "v": velo,
            "r": 0
        }
        response = self._send_and_receive()
        
        if response and response.get("trajectory_state"):
            print(f"MoveJ_P OK!\t{response}")
        else:
            print(f"ERROR! MoveJ_P False!\t{response}")
            sys.exit(0)

    def servo_j(self, joints, follow):
        self.command_msg = {
            "command": "movej_canfd",
            "joint": [int(1000 * j * 180/np.pi) for j in joints]
        }
        response = self._send_and_receive()
        
        if response and response.get("arm_err") == 0:
            pass  # ServoJ OK
        else:
            print(f"WARNING! ServoJ False!\t{response}")

    def _send_and_receive(self):
        try:
            cmd_str = json.dumps(self.command_msg) + "\r\n"
            self.rlm_socket.send(cmd_str.encode())
            
            recv_times = 0
            while recv_times < 3:
                data = self.rlm_socket.recv(1000)
                if len(data) >= 10:
                    try:
                        self.return_msg = json.loads(data.decode())
                        if not self.return_msg:
                            print(f"WARNING! Missing a return message: {data.decode()}")
                        return self.return_msg
                    except json.JSONDecodeError:
                        print(f"WARNING! Invalid JSON received: {data.decode()}")
                recv_times += 1
                
            if recv_times == 3:
                print("ERROR! Can't receive message!")
                sys.exit(0)
                
        except Exception as e:
            print(f"ERROR! Communication error: {e}")
            sys.exit(0)

if __name__ == "__main__":
    # rm_command = RMCommand()
    # rm_command.connect_tcp_socket()
    # pose,joint = rm_command.read_arm_state()
    # print(pose)
    # print(joint)
    # # convert joint from degree to radian
    # joint = [j*np.pi/180 for j in joint]
    # print(fk(joint))
    #  # [88.442, -34.931, -27.115, 2.156, 10.255, 98.338, -122.639]
    # # rm_command.move_j([88.437, -34.931, -27.111, 2.16, 10.254, 98.338, -122.639], 5)
    # # rm_command.move_j([0,0,0,0,0,0,0], 5)
    # rm_command.rlm_socket.close()
    rm_command = RMCommand()
    rm_command.connect_tcp_socket()
    # rm_command.move_j([88.437, -34.931, -27.111, 2.16, 10.254, 98.338, -122.639], 5)
    
    while True:
        # print("start")
        joint = rm_command.read_j()
        joint_rad = [j*np.pi/180 for j in joint]
        # print(joint_rad)
        # joint_rad = [j*np.pi/180 for j in joint]
        ee2base = fk(joint_rad) # 可
        print(ee2base)
        time.sleep(1)