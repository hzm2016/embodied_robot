# PPCA2 实时姿态估计与碰撞检测
# 机械臂姿态规避
## 所属文件夹
\PPCAlib  
main_collision.py

## 使用方法
```
python PPCAlib\realsensevideo.py 
```  
标定机械臂base位置。  
并根据此手动更新 PPCAlib\camera_manager.py中参数。  
```
python src\main_collision.py
```
# 机械臂末端视觉伺服跟踪
## 所属文件夹
\RCM  
main_track.py

## 使用方法
```
python src\main_track.py
```
## 参考与致谢
- YOLO/Ultralytics: https://github.com/ultralytics/ultralytics
- Intel RealSense SDK: https://github.com/IntelRealSense/librealsense
- PyTorch: https://pytorch.org/

---

## 许可证
仅供研究与教学用途。若用于商业目的，请自行评估并遵循各第三方依赖的许可证条款。
