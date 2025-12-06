"""
主程序入口
实时姿态估计与碰撞检测系统
"""

import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO


def main():
    # 初始化 RealSense 相机
    pipeline = rs.pipeline()
    config = rs.config()
    
    # 配置彩色流
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    
    # 启动相机
    print("正在启动 RealSense 相机...")
    try:
        pipeline.start(config)
        print("RealSense 相机已启动")
    except Exception as e:
        print(f"启动相机失败: {e}")
        return
    
    # 加载 YOLO 姿态估计模型
    print("正在加载 YOLO 模型...")
    model = YOLO("yolo11n-pose.pt")
    print("模型加载完成")
    
    try:
        while True:
            # 等待并获取帧
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            
            if not color_frame:
                continue
            
            # 转换为 numpy 数组
            frame = np.asanyarray(color_frame.get_data())
            
            # 使用 YOLO 进行姿态估计
            results = model(frame, verbose=False)
            
            # 在帧上绘制关键点和骨架
            annotated_frame = results[0].plot()
            
            # 显示结果
            cv2.imshow('RealSense - YOLO Pose Detection', annotated_frame)
            
            # 按 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\n程序被用户中断")  
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        # 停止相机并关闭窗口
        pipeline.stop()
        cv2.destroyAllWindows()
        print("程序已退出")


if __name__ == "__main__":
    main()