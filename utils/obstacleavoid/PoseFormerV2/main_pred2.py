
from PPCAlib.realtime_system_with_prediction import RealtimePoseCollisionPredictionSystem



def main():
    """主函数"""
    print("Enhanced Realtime Pose Collision and Prediction System")
    print("=" * 60)
    capsule_radius = 0.05 
    # 可以通过命令行参数控制是否启用预测
    import sys
    enable_prediction = True
    if len(sys.argv) > 1 and sys.argv[1] == '--no-prediction':
        enable_prediction = False
        print("Motion prediction disabled")
    
    try:
        system = RealtimePoseCollisionPredictionSystem(
            cube_position=[0.0, 0.0, 0.0],
            capsule_radius=capsule_radius,
            enable_prediction=enable_prediction
        )
        system.initialize()
        system.run()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()