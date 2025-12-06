"""
基于pose_3d_10输入的动作预测类
模仿realtime_prediction_time_fast2 copy.py，但输入改为realtime_system.py中的pose_3d_10
输出格式为(T,N,3)，其中N=10
"""

import torch
import numpy as np
import time
import yaml
import argparse
from collections import deque

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sub_dir = os.path.join(project_root, 'AuxFormerV1')
sys.path.append(sub_dir)
from AuxFormerV1.model.model import AuxFormer
import warnings
warnings.filterwarnings("ignore")


class MotionPredictor:
    """基于pose_3d_10的动作预测类"""
    
    def __init__(self, config_path='AuxFormerV1/cfg/h36m_short.yml', model_path='AuxFormerV1/ckpt/10h36m_ckpt_best.pth.tar'):
        """
        初始化动作预测器
        
        Args:
            config_path: 配置文件路径
            model_path: 模型权重文件路径
        """
        # 加载配置
        self.args = self._load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化AuxFormer模型
        self.model = self._load_auxformer_model(model_path)
        
        # 存储历史帧的队列（存储past_length帧历史数据）
        self.pose_history = deque(maxlen=self.args.past_length)
        
        # 用于存储预测结果
        self.predicted_future = None
        
        # 10关节的骨骼连接定义（与visual_pybullet.py保持一致）
        self.bone_connections = [
            (0, 1), (1, 2), (2, 3),  # 躯干连接
            (1, 4), (4, 5), (5, 6),  # 左臂连接  
            (1, 7), (7, 8), (8, 9)   # 右臂连接
        ]
        
        print(f"动作预测器初始化完成")
        print(f"设备: {self.device}")
        print(f"历史帧长度: {self.args.past_length}")
        print(f"预测帧长度: {self.args.future_length}")
        print(f"关节数量: {self.args.n_agent}")
    
    def _load_config(self, config_path):
        """加载配置文件"""
        try:
            with open(config_path, 'r') as f:
                yml_arg = yaml.load(f, Loader=yaml.FullLoader)
            
            # 创建参数对象
            parser = argparse.ArgumentParser()
            parser.set_defaults(**yml_arg)
            args = parser.parse_args([])  # 空参数列表，使用默认值
            
            return args
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            # 返回默认配置
            return self._get_default_config()
    
    def _get_default_config(self):
        """获取默认配置"""
        parser = argparse.ArgumentParser()
        # 设置默认参数（基于h36m_short.yml的典型值）
        default_config = {
            'past_length': 10,
            'future_length': 10,
            'n_agent': 10,
            'nf': 256,
            'mask_ratio': 0.5,
            'decoder_dim': 256,
            'encoder_depth': 6,
            'decoder_depth': 6,
            'dim_per_head': 64,
            'same_head': True,
            'range_mask_ratio': 0.0,
            'mlp_head': True,
            'mask_past': False,
            'mask_range': False,
            'multi_output': False,
            'decoder_masking': False,
            'pred_all': False,
            'mlp_dim': 512,
            'noise_dev': 0.0,
            'part_noise': False,
            'denoise_mode': 'raw',
            'part_noise_ratio': 0.0,
            'add_joint_token': False,
            'concat_vel': False,
            'only_recons_past': False,
            'add_residual': False,
            'denoise': False,
            'regular_masking': False,
            'multi_same_head': False,
            'range_noise_dev': 0.0
        }
        parser.set_defaults(**default_config)
        return parser.parse_args([])
    
    def _load_auxformer_model(self, model_path):
        """加载预训练的AuxFormer模型"""
        try:
            model = AuxFormer(
                in_dim=3,  # 改为3维输入（直接使用3D坐标）
                h_dim=self.args.nf, 
                past_timestep=self.args.past_length,
                future_timestep=self.args.future_length, 
                mask_ratio=self.args.mask_ratio, 
                decoder_dim=self.args.decoder_dim, 
                num_heads=8, 
                encoder_depth=self.args.encoder_depth,
                decoder_depth=self.args.decoder_depth,
                decoder_dim_per_head=self.args.dim_per_head,
                same_head=self.args.same_head,
                range_mask_ratio=self.args.range_mask_ratio,
                mlp_head=self.args.mlp_head,
                mask_past=self.args.mask_past,
                mask_range=self.args.mask_range,
                multi_output=self.args.multi_output,
                decoder_masking=self.args.decoder_masking,
                pred_all=self.args.pred_all,
                mlp_dim=self.args.mlp_dim,
                dim_per_head=self.args.dim_per_head,
                noise_dev=self.args.noise_dev,
                part_noise=self.args.part_noise,
                denoise_mode=self.args.denoise_mode,
                part_noise_ratio=self.args.part_noise_ratio,
                add_joint_token=self.args.add_joint_token,
                n_agent=self.args.n_agent,
                concat_vel=self.args.concat_vel,
                only_recons_past=self.args.only_recons_past,
                add_residual=self.args.add_residual,
                denoise=self.args.denoise,
                regular_masking=self.args.regular_masking,
                multi_same_head=self.args.multi_same_head,
                range_noise_dev=self.args.range_noise_dev
            ).to(self.device)
            
            # 加载预训练权重
            print(f'尝试加载模型: {model_path}')
            model_ckpt = torch.load(model_path, map_location=self.device)
            model.load_state_dict(model_ckpt['state_dict'])
            model.eval()
            print("模型加载成功")
            
            return model
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            print("将使用未预训练的模型")
            
            # 如果加载失败，返回未预训练的模型
            model = AuxFormer(
                in_dim=3,
                h_dim=256,
                past_timestep=10,
                future_timestep=10,
                mask_ratio=0.5,
                decoder_dim=256,
                num_heads=8,
                encoder_depth=6,
                decoder_depth=6,
                decoder_dim_per_head=64,
                same_head=True,
                range_mask_ratio=0.0,
                mlp_head=True,
                mask_past=False,
                mask_range=False,
                multi_output=False,
                decoder_masking=False,
                pred_all=False,
                mlp_dim=512,
                dim_per_head=64,
                noise_dev=0.0,
                part_noise=False,
                denoise_mode='raw',
                part_noise_ratio=0.0,
                add_joint_token=False,
                n_agent=10,
                concat_vel=False,
                only_recons_past=False,
                add_residual=False,
                denoise=False,
                regular_masking=False,
                multi_same_head=False,
                range_noise_dev=0.0
            ).to(self.device)
            model.eval()
            return model
    
    def add_pose(self, pose_3d_10):
        """
        添加新的姿态到历史队列
        
        Args:
            pose_3d_10: (10, 3) numpy array, 10个关节的3D坐标
        """
        if pose_3d_10 is None:
            return
            
        # 确保输入格式正确
        pose_3d_10 = np.array(pose_3d_10)
        if pose_3d_10.shape != (10, 3):
            print(f"警告: pose_3d_10形状应为(10, 3)，当前为{pose_3d_10.shape}")
            return
        
        # 添加到历史队列
        self.pose_history.append(pose_3d_10.copy())
    
    def transform_pose(self, past_poses, back = False):
        K = 5
        if len(self.pose_history) == 0:
            return
        if back == False:
            past_poses[:,:,[1,2]] = past_poses[:,:,[2,1]]
            past_poses = K * past_poses
            offset = np.array([0., 5., 0.]) - past_poses[0,1,:]
            past_poses += offset
        else:
            offset = np.array([0., -5., 0.]) + past_poses[0,1,:]
            past_poses += offset
            past_poses = past_poses / K
            past_poses[:,:,[1,2]] = past_poses[:,:,[2,1]]
        
        return past_poses


    def predict_future_motion(self):
        """
        预测未来动作
        
        Returns:
            predicted_motion: (T, N, 3) numpy array, 其中T=future_length, N=10
                             如果历史数据不足，返回None
        """
        if len(self.pose_history) < self.args.past_length:
            return None
            
        try:
            # 将历史数据转换为模型输入格式
            # pose_history: list of (10, 3) -> (1, 10, past_length, 3)
            past_poses = np.stack(list(self.pose_history), axis=0)  # (past_length, 10, 3)
            past_poses = self.transform_pose(past_poses, back=False)
            past_poses = past_poses.transpose(1, 0, 2)  # (10, past_length, 3)
            past_poses = torch.FloatTensor(past_poses).unsqueeze(0).to(self.device)  # (1, 10, past_length, 3)
            
            with torch.no_grad():
                # 创建虚假的未来数据用于预测
                future_fake = torch.zeros(1, self.args.n_agent, self.args.future_length, 3).to(self.device)
                all_traj = torch.cat([past_poses, future_fake], dim=2)  # (1, 10, past_length+future_length, 3)

                # 创建mask（假设所有历史数据都是可见的）
                unmask_ind = torch.ones(1, self.args.n_agent, self.args.past_length, dtype=torch.bool).to(self.device)
                # 为future_length部分填充False（未来时间步都是masked的）
                future_mask = torch.zeros(1, self.args.n_agent, self.args.future_length, dtype=torch.bool).to(self.device)
                unmask_ind = torch.cat([unmask_ind, future_mask], dim=2)  # (1, 10, past_length+future_length)
                # 进行预测
                if hasattr(self.model, 'predict_with_mask'):
                    loc_pred = self.model.predict_with_mask(all_traj, unmask_ind)  # (1, 10, future_length, 3)
                else:
                    # 如果没有predict_with_mask方法，使用forward方法
                    loc_pred = self.model(all_traj, unmask_ind)  # (1, 10, future_length, 3)
                
                # 转换为所需的输出格式 (T, N, 3)
                predicted_motion = loc_pred[0].cpu().numpy()  # (10, future_length, 3)
                predicted_motion = predicted_motion.transpose(1, 0, 2)  # (future_length, 10, 3)
                predicted_motion = self.transform_pose(predicted_motion, back=True)
                self.predicted_future = predicted_motion
                return predicted_motion
                
        except Exception as e:
            print(f"预测过程中出错: {e}")
            return None
    
    def get_current_pose(self):
        """
        获取当前最新的姿态
        
        Returns:
            current_pose: (10, 3) numpy array 或 None
        """
        if len(self.pose_history) > 0:
            return self.pose_history[-1].copy()
        return None
    
    def get_prediction_status(self):
        """
        获取预测状态信息
        
        Returns:
            status: dict 包含状态信息
        """
        return {
            'history_length': len(self.pose_history),
            'required_length': self.args.past_length,
            'can_predict': len(self.pose_history) >= self.args.past_length,
            'future_length': self.args.future_length,
            'has_prediction': self.predicted_future is not None
        }
    
    def reset(self):
        """重置预测器状态"""
        self.pose_history.clear()
        self.predicted_future = None
        print("动作预测器已重置")


class MotionPredictorWithVisualization:
    """
    集成PyBullet可视化的动作预测器
    """
    
    def __init__(self, config_path='cfg/h36m_short.yml', model_path='ckpt/10h36m_ckpt_best.pth.tar'):
        """
        初始化预测器和可视化
        
        Args:
            config_path: 配置文件路径
            model_path: 模型权重文件路径
        """
        from visual_pybullet import RobotArmVisualizer
        
        # 初始化动作预测器
        self.predictor = MotionPredictor(config_path, model_path)
        
        # 初始化PyBullet可视化
        self.visualizer = RobotArmVisualizer()
        
        # 预测结果的可视化参数
        self.prediction_colors = [
            [1, 0, 0, 0.3],  # 红色，透明度0.3
            [1, 0.5, 0, 0.4],  # 橙色，透明度0.4
            [1, 1, 0, 0.5],  # 黄色，透明度0.5
            [0, 1, 0, 0.6],  # 绿色，透明度0.6
            [0, 0, 1, 0.7],  # 蓝色，透明度0.7
        ]
        
        # 存储预测帧的可视化对象
        self.prediction_frames = []
        
        print("集成可视化的动作预测器初始化完成")
    
    def update_and_predict(self, pose_3d_10):
        """
        更新姿态并进行预测，同时更新可视化
        
        Args:
            pose_3d_10: (10, 3) numpy array, 10个关节的3D坐标
        
        Returns:
            predicted_motion: (T, N, 3) numpy array 或 None
        """
        # 添加新姿态
        self.predictor.add_pose(pose_3d_10)
        
        # 更新当前姿态可视化
        if pose_3d_10 is not None:
            self.visualizer.set_human_model(pose_3d_10)
        
        # 进行预测
        predicted_motion = self.predictor.predict_future_motion()
        
        # 可视化预测结果
        self._visualize_prediction(predicted_motion)
        
        # 执行仿真步进
        self.visualizer.step_simulation()
        
        return predicted_motion
    
    def _visualize_prediction(self, predicted_motion):
        """
        可视化预测的未来动作
        
        Args:
            predicted_motion: (T, N, 3) numpy array
        """
        # 清除之前的预测可视化
        self._clear_prediction_frames()
        
        if predicted_motion is None:
            return
        
        # 选择几个关键帧进行可视化
        T = predicted_motion.shape[0]
        if T > 0:
            # 选择最多5个关键帧进行可视化
            step = max(1, T // 5)
            selected_frames = range(0, T, step)[:5]
            
            for i, frame_idx in enumerate(selected_frames):
                if frame_idx < T:
                    future_pose = predicted_motion[frame_idx]  # (10, 3)
                    color = self.prediction_colors[i % len(self.prediction_colors)]
                    
                    # 创建预测帧的可视化（这里需要扩展visual_pybullet.py的功能）
                    # 目前只显示当前帧和最后一帧
                    if frame_idx == T - 1:  # 只显示最后一帧预测
                        self._create_prediction_frame(future_pose, color)
    
    def _create_prediction_frame(self, pose, color):
        """
        创建预测帧的可视化
        
        Args:
            pose: (10, 3) numpy array
            color: [r, g, b, alpha] 颜色
        """
        # 这里可以扩展visual_pybullet.py来支持多个人体骨骼的显示
        # 目前简化处理，可以考虑创建半透明的骨骼
        pass
    
    def _clear_prediction_frames(self):
        """清除预测帧的可视化"""
        # 清除之前的预测可视化对象
        for frame_obj in self.prediction_frames:
            # 这里需要PyBullet的清除对象功能
            pass
        self.prediction_frames.clear()
    
    def get_status(self):
        """获取预测器状态"""
        return self.predictor.get_prediction_status()
    
    def reset(self):
        """重置预测器"""
        self.predictor.reset()
        self._clear_prediction_frames()
    
    def cleanup(self):
        """清理资源"""
        self.visualizer.disconnect()


# 测试函数
def test_motion_predictor():
    """测试动作预测器"""
    print("开始测试动作预测器...")
    
    # 创建预测器（使用默认配置，可能没有预训练模型）
    try:
        predictor = MotionPredictor()
    except:
        print("无法加载预训练模型，使用默认配置创建预测器")
        predictor = MotionPredictor(config_path='invalid_path')
    
    # 生成测试数据
    print("生成测试姿态数据...")
    test_poses = []
    for i in range(15):  # 生成15帧数据
        # 创建简单的动作序列
        t = i * 0.1
        base_pose = np.array([
            [0, 1, 1.7],    # 头部
            [0, 1, 1.4],    # 颈部/肩部中心
            [0, 1, 1.0],    # 躯干中部
            [0, 1, 0.5],    # 臀部
            [-0.2, 1, 1.3], # 左肩
            [-0.4, 1, 1.0], # 左肘
            [-0.6, 1, 0.7], # 左手
            [0.2, 1, 1.3],  # 右肩
            [0.4, 1, 1.0],  # 右肘
            [0.6, 1, 0.7],  # 右手
        ])
        
        # 添加简单的摆动
        base_pose[5][0] += 0.2 * np.sin(t * 2)  # 左肘摆动
        base_pose[8][0] += 0.2 * np.cos(t * 2)  # 右肘摆动
        
        test_poses.append(base_pose)
    
    # 测试预测过程
    print("测试预测过程...")
    for i, pose in enumerate(test_poses):
        predictor.add_pose(pose)
        status = predictor.get_status()
        print(f"帧 {i+1}: 历史长度 {status['history_length']}/{status['required_length']}, 可预测: {status['can_predict']}")
        
        if status['can_predict']:
            predicted = predictor.predict_future_motion()
            if predicted is not None:
                print(f"  预测成功: 输出形状 {predicted.shape}")
            else:
                print("  预测失败")
    
    print("测试完成")


if __name__ == "__main__":
    test_motion_predictor()