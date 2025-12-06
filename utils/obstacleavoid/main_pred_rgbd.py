"""
主程序入口
实时姿态估计与碰撞检测系统
"""

from PPCAlib.realtime_system import RealtimePoseCollisionSystem


def main():
    """主函数"""
    # 设置立方体参数
    cube_position = [0., 0.0, 0.0]  # 立方体位置
    capsule_radius = 0.05           # 胶囊半径
    cube_size = 0.4                 # 立方体大小
    
    # 初始化系统
    system = RealtimePoseCollisionSystem(
        robot_base_position=cube_position,
        capsule_radius=capsule_radius,
        cube_size=cube_size,
        enable_prediction=True,
        use_rgbd=True,
        Realrobot=False
    )
    
    try:
        # 初始化所有组件
        system.initialize()
        
        # 运行主循环
        system.run()
        
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 确保资源被正确释放
        system.cleanup()


if __name__ == "__main__":
    main()