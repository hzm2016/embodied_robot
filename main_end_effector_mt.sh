#!/bin/bash

# run end-effector conde
cd /home/p9/Projects/embodiedrobot/endeffector_ctrl/  
source /home/p9/anaconda3/etc/profile.d/conda.sh  
conda activate p9
sudo /home/p9/anaconda3/envs/p9/bin/python main_mt.py 