#!/usr/bin/env python3
"""
使用示例：演示如何使用统一的实时姿态检测系统
支持可选的动作预测功能和RGBD相机
"""

from PPCAlib.realtime_system import RealtimePoseCollisionSystem

def main():
    """主函数 - 演示如何使用统一系统"""
    
    print("=== 实时姿态检测与碰撞避免系统 ===")
    print("选择相机类型:")
    print("1. RGB相机 (普通摄像头)")
    print("2. RGBD相机 (RealSense L515)")
    
    while True:
        try:
            camera_choice = input("请输入选择 (1/2): ").strip()
            if camera_choice in ['1', '2']:
                break
            else:
                print("请输入 1 或 2")
        except KeyboardInterrupt:
            print("\n程序退出")
            return
    
    use_rgbd = (camera_choice == '2')
    
    print("\n选择运行模式:")
    print("1. 基础模式 (无动作预测)")
    print("2. 增强模式 (包含动作预测)")
    
    while True:
        try:
            choice = input("请输入选择 (1/2): ").strip()
            if choice in ['1', '2']:
                break
            else:
                print("请输入 1 或 2")
        except KeyboardInterrupt:
            print("\n程序退出")
            return
    
    # 根据选择确定是否启用预测
    enable_prediction = (choice == '2')
    
    # 系统配置
    cube_position = [0.0, 1.0, 1.0]  # 立方体初始位置
    capsule_radius = 0.1  # 人体胶囊半径
    cube_size = 0.5  # 立方体大小
    
    print(f"\n初始化系统...")
    print(f"相机类型: {'RGBD (RealSense L515)' if use_rgbd else 'RGB (普通摄像头)'}")
    print(f"动作预测: {'启用' if enable_prediction else '禁用'}")
    
    if use_rgbd:
        print("\n注意: RGBD模式下，3D姿态将直接从深度信息计算，精度更高！")
    
    # 创建系统实例
    system = RealtimePoseCollisionSystem(
        cube_position=cube_position,
        capsule_radius=capsule_radius,
        cube_size=cube_size,
        enable_prediction=enable_prediction,
        use_rgbd=use_rgbd
    )
    
    try:
        # 初始化系统
        system.initialize()
        
        # 运行主循环
        system.run()
        
    except KeyboardInterrupt:
        print("\n用户中断程序")
    except Exception as e:
        print(f"\n系统错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理资源
        system.cleanup()
        print("程序结束")

if __name__ == "__main__":
    main()