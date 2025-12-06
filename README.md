There are four modules, robot-arm-ctrl, end-effector-ctrl, visual_module, and voice_module. 

Two control loop are run for robot-arm and end-effector. 

Global robot state : joint angle, joint velocity 

Zero force :  

ssh root@192.168.1.136 password: root 
cd /opt/ecat_test ./robot_control_2 final.xml

sudo /home/p9/anaconda3/envs/p9/bin/python reset_all_motors.py

sudo /home/p9/anaconda3/envs/p9/bin/python main_mt.py