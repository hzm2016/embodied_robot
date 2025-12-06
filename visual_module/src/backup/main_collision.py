"""
主程序入口
实时姿态估计与碰撞检测系统
"""
import os
import sys
from typing import Any, Dict

try:
    import yaml
except ImportError as exc:
    raise RuntimeError("PyYAML is required. Install it with 'pip install pyyaml'.") from exc


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CONFIG_PATH = os.path.join(ROOT_DIR, 'config', 'realtime_system.yml')
sys.path.append(ROOT_DIR)  

from PPCAlib.realtime_system import RealtimePoseCollisionSystem


def load_config(config_path: str = CONFIG_PATH) -> Dict[str, Any]:
    """加载YAML配置，如果缺失则返回空配置"""
    full_path = os.path.abspath(config_path)
    if not os.path.exists(full_path):
        print(f"Config file not found at {full_path}. Using default parameters.")
        return {}

    with open(full_path, 'r', encoding='utf-8') as config_file:
        data = yaml.safe_load(config_file) or {}
    return data


def main():  
    """主函数"""  
    config = load_config()    
    system_cfg = config.get('realtime_system', {})  

    cube_position = system_cfg.get('robot_base_position', [0.0, 0.0, 0.0])
    capsule_radius = system_cfg.get('capsule_radius', 0.05)
    cube_size = system_cfg.get('cube_size', 0.4)
    enable_prediction = system_cfg.get('enable_prediction', False)
    realrobot = system_cfg.get('realrobot', False)  

    # 初始化系统
    system = RealtimePoseCollisionSystem(
        robot_base_position=cube_position,
        capsule_radius=capsule_radius,
        cube_size=cube_size,
        enable_prediction=enable_prediction,
        Realrobot=realrobot
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