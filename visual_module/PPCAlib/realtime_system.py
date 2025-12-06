"""
实时姿态估计与碰撞检测系统模块（支持可选动作预测功能）
主控制器，集成所有组件
使用YOLO进行人体姿态估计
"""  
import cv2
import numpy as np  
from math import pi
import time  

from .model_manager import ModelManager
from .pose_estimator import PoseEstimator
from .collision_visualizer import CollisionVisualizer
from .camera_manager import CameraManager
from .joint_mapper import JointMapper  
from .human_collision_detector import HumanCollisionDetector  

from .control import ControlSystem   
# from .visual_pybullet import RobotArmVisualizer    
from Robotic_Arm.rm_robot_interface import *    



class RealtimePoseCollisionSystem:
    """实时姿态估计与碰撞检测系统 - 支持可选动作预测功能和RGBD相机的主控制器"""
    
    def __init__(self, robot_base_position=None, capsule_radius=0.1, cube_size=0.5, 
                 enable_prediction=False, Realrobot=False):  
        
        # 基础组件  
        self.model_manager = ModelManager() 
        self.pose_estimator = None 
        self.visualizer = CollisionVisualizer()  
        self.camera_manager = CameraManager() 
        self.joint_mapper = JointMapper()  
        self.collision_detector = HumanCollisionDetector(
            capsule_radius=capsule_radius, 
            cube_size=cube_size, 
            base_position=robot_base_position
        ) 
        self.control_system = ControlSystem()  
        
        # 动作预测相关组件
        self.enable_prediction = enable_prediction   
        self.motion_predictor = None   
        if self.enable_prediction:  
            try:
                from .motion_predictor import MotionPredictor
                self.motion_predictor = MotionPredictor()
                print("动作预测器初始化成功")
            except Exception as e:
                print(f"动作预测器初始化失败: {e}")
                print("将禁用动作预测功能")
                self.enable_prediction = False  
        
        # 预测相关参数
        self.show_prediction = True
        self.prediction_frames = 5  # 显示的预测帧数
        self.Realrobot = Realrobot   
        
        if self.Realrobot:  
            self.Realman = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
            handle = self.Realman.rm_create_robot_arm("192.168.2.18", 8080, 3)  

    def initialize(self):  
        """初始化系统"""
        print("Initializing Realtime Pose Estimation and Collision Detection System...")

        # 加载模型
        self.model_manager.load_all_models()
        self.pose_estimator = PoseEstimator(self.model_manager)

        # 初始化摄像头
        self.camera_manager.initialize()

        print("System initialized successfully!")
        print("\nFeatures enabled:")
        print("  ✓ Real-time 2D pose detection (YOLO)")
        print("  ✓ 3D pose estimation")
        print("  ✓ Collision detection")
        print("  ✓ Robot control")
        print(f"  {'✓' if self.enable_prediction else '✗'} Motion prediction")
        print("  Camera: RGBD (L515)")

        print("\nControls:")
        print("  'q' - Quit")
        print("  'r' - Reset cube position")
        print("  'c' - Change cube size")
        if self.enable_prediction:
            print("  'p' - Toggle prediction display")
        print("  SPACE - Pause/Resume")

        if self.enable_prediction:
            status = self.motion_predictor.get_prediction_status()
            print(f"\nPrediction Settings:")
            print(f"  History length: {status['required_length']} frames")
            print(f"  Future length: {status['future_length']} frames")
            print(f"  Display frames: {self.prediction_frames}")
        
    def run(self):   
        """运行主循环"""  
        frame_count = 0  
        paused = False  
        theta = [0., pi/4, 0., pi/2, 0., -pi/4, 0.]    

        if self.Realrobot:     
            # self.Realman.rm_movej(np.rad2deg(theta),10,0,0,1)
            arm_data = self.Realman.rm_get_arm_current_trajectory()
            print("now_degrees:",arm_data["data"])  
            theta = np.deg2rad(arm_data["data"])   

        self.control_system.get_ori_joint(theta)   
            
        print("\nStarting main loop...")    
        if self.enable_prediction:   
            print("Note: Prediction will start after collecting enough history frames")   

        while True:
            if not paused:
                time_start = time.time()
                ret, frame, depth_frame, depth_intrinsics = self.camera_manager.read_rgbd_frame()
                if not ret:
                    print("Error: Could not read RGBD frame.")
                    break
                
                frame_count += 1   
                
                # 获取2D关键点与可视化帧
                input_2D_no, input_2D_sc, annotated_frame = self.pose_estimator.get_2d_keypoints(frame)
                  
                # 估计3D姿态（仅RGBD模式）  
                pose_3d_17, confidence = self.pose_estimator.estimate_3d_pose_from_rgbd(
                    input_2D_no, input_2D_sc, depth_frame, depth_intrinsics, frame.shape)
                    
                # 映射到10关节用于碰撞检测
                try:
                    pose_3d_10 = self.joint_mapper.map_joints(pose_3d_17)
                    
                    # 动作预测（如果启用）
                    predicted_motion = None
                    pose_3d_pred = None
                    future_collision_risk = None
                    
                    if self.enable_prediction and pose_3d_10 is not None:
                        self.motion_predictor.add_pose(pose_3d_10)
                        predicted_motion = self.motion_predictor.predict_future_motion()
                        
                        if predicted_motion is not None:
                            # 适合真实位置的坐标（预测数据）
                            offset2 = self.camera_manager.robot_camera_coord()
                            predicted_motion += offset2
                            pose_3d_pred = predicted_motion[[3, 6], :, :]
                    
                    # 适合真实位置的坐标（当前数据）
                    offset = self.camera_manager.robot_camera_coord()
                    pose_3d_10 += offset
                    
                    # 更新碰撞检测器
                    self.collision_detector.set_current_joints(pose_3d_10)
                    self.collision_detector.update_human_pose(pose_3d_10)
                    self.collision_detector.update_robotic_pose(theta)
                    print(theta)
                    
                    # 执行碰撞检测
                    collision_result = self.collision_detector.check_collision()
                    
                    # 分析未来碰撞风险（如果启用预测）
                    if self.enable_prediction and predicted_motion is not None:
                        future_collision_risk = self._analyze_future_collision_risk(
                            predicted_motion, theta, collision_result)
                    
                except Exception as e:
                    print(f"Processing error: {e}")
                    collision_result = {
                        'is_collision': False,
                        'min_distance': float('inf'),
                        'nearest_points': None,
                        'collision_points': None
                    }
                    pose_3d_10 = None
                    predicted_motion = None
                    pose_3d_pred = None
                    future_collision_risk = None
                
                # 只有检测到人体时才进行控制 
                if np.mean(input_2D_sc)>0.3: 
                    theta = self.control_system.update(collision_result, theta)   

                    if self.Realrobot:
                        # self.Realman.rm_movej(np.rad2deg(theta),10,0,0,0)
                        print("now_degrees:",np.rad2deg(theta))
                        self.Realman.rm_movej_canfd(np.rad2deg(theta),False,0,1,50)
                else:
                    print("No human detected - Robot control paused")

                # 3D姿态，消耗很大
                if not self.Realrobot:
                    self.visualizer.draw_collision_detection(self.collision_detector, pose_3d_10, collision_result) 
                
                # 在2D图像上显示状态信息  
                self._draw_status_info(annotated_frame, collision_result, 
                                     predicted_motion, future_collision_risk, frame_count)
                
                # 显示2D摄像头画面
                window_title = 'RealSense - YOLO Pose Detection'
                cv2.imshow(window_title, annotated_frame)

                time_end = time.time()
                print("time:",time_end - time_start)
            
            # 处理键盘输入  
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):    
                break  
            elif key == ord(' '):                             # 空格键暂停/恢复
                paused = not paused
                print("Paused" if paused else "Resumed")
            elif key == ord('p') and self.enable_prediction:  # 切换预测显示
                self.show_prediction = not self.show_prediction
                print(f"Prediction display: {'ON' if self.show_prediction else 'OFF'}")
                # if not self.show_prediction:
                #     # 清除预测显示
                #     self.robot.set_predicted_models(None)
        
        self.cleanup()
    
    def _analyze_future_collision_risk(self, predicted_motion, current_theta, current_collision):
        """
        分析未来碰撞风险
        
        Args:
            predicted_motion: (T, N, 3) 预测的人体动作
            current_theta: 当前机器人关节角度
            current_collision: 当前碰撞检测结果
            
        Returns:
            risk_info: dict 包含风险分析信息
        """
        if predicted_motion is None:
            return None
        
        try:
            # 简单的风险分析：检查预测动作中是否有关节点靠近危险区域
            risk_frames = []
            min_distances = []
            
            for t in range(predicted_motion.shape[0]):
                future_pose = predicted_motion[t]  # (10, 3)
                
                # 更新人体姿态到碰撞检测器
                self.collision_detector.update_human_pose(future_pose)
                
                # 检查未来碰撞
                future_collision = self.collision_detector.check_collision()
                min_distances.append(future_collision['min_distance'])
                
                if future_collision['is_collision']:
                    risk_frames.append(t)
            
            # 计算风险等级
            min_future_distance = min(min_distances) if min_distances else float('inf')
            risk_level = "LOW"
            if min_future_distance < 0.3:
                risk_level = "HIGH"
            elif min_future_distance < 0.5:
                risk_level = "MEDIUM"
            
            return {
                'risk_level': risk_level,
                'collision_frames': risk_frames,
                'min_future_distance': min_future_distance,
                'trajectory_length': predicted_motion.shape[0]
            }
            
        except Exception as e:
            print(f"Future collision analysis error: {e}")
            return None
    
    def _draw_status_info(self, frame, collision_result, predicted_motion, 
                         future_risk, frame_count):
        """在2D图像上绘制状态信息"""
        h, w = frame.shape[:2]
        
        # 当前碰撞状态
        if collision_result['is_collision']:
            text = "COLLISION DETECTED!"
            color = (0, 0, 255)  # 红色
        else:
            text = f"Safe - Distance: {collision_result['min_distance']:.3f}m"
            color = (0, 255, 0)  # 绿色
        
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, color, 2, cv2.LINE_AA)
        
        # 预测状态（仅在启用时显示）
        y_offset = 60
        if self.enable_prediction:
            status = self.motion_predictor.get_prediction_status()
            pred_text = f"Prediction: {status['history_length']}/{status['required_length']}"
            if predicted_motion is not None:
                pred_text += f" -> {predicted_motion.shape[0]} frames"
                pred_color = (0, 255, 255)  # 黄色
            else:
                pred_text += " (collecting data...)"
                pred_color = (128, 128, 128)  # 灰色
            
            cv2.putText(frame, pred_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, pred_color, 1, cv2.LINE_AA)
            y_offset += 25
            
            # 未来风险信息
            if future_risk is not None:
                risk_text = f"Future Risk: {future_risk['risk_level']}"
                if future_risk['collision_frames']:
                    risk_text += f" (frames: {len(future_risk['collision_frames'])})"
                
                risk_color = (0, 255, 0)  # 绿色
                if future_risk['risk_level'] == "HIGH":
                    risk_color = (0, 0, 255)  # 红色
                elif future_risk['risk_level'] == "MEDIUM":
                    risk_color = (0, 165, 255)  # 橙色
                
                cv2.putText(frame, risk_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, risk_color, 1, cv2.LINE_AA)
                y_offset += 25
        
        # 显示控制说明
        if frame_count < 150:  # 前5秒显示控制说明
            controls = [
                f"Controls: 'q'-quit{', ''p''-toggle prediction' if self.enable_prediction else ''}",
                "'r'-reset cube, 'c'-change size, SPACE-pause"
            ]
            for i, control_text in enumerate(controls):
                cv2.putText(frame, control_text, (10, h - 40 + i * 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    
    def cleanup(self):  
        """清理资源"""
        self.camera_manager.release()
        cv2.destroyAllWindows()
        self.visualizer.cleanup()
        # if hasattr(self, 'robot'):
        #     self.robot.disconnect()
        print("System ended.")  