"""
测试 pose_estimator.get_2d_keypoints() 方法
用于诊断在其他电脑上运行失败的问题

使用方法:
    python test_2d_keypoints.py

这个测试会逐步检查所有可能的失败点:
1. 基础库导入 (torch, opencv, numpy)
2. 项目模块导入 (ModelManager, PoseEstimator)
3. 模型文件加载
4. get_2d_keypoints 方法执行

如果出现错误，会精确定位到具体的代码行和错误原因。
"""

import sys
import os
import traceback

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def test_imports():
    """测试所有必要的导入"""
    print("=" * 60)
    print("第1步: 测试基础导入")
    print("=" * 60)
    
    print("[DEBUG] 开始测试 torch 导入...")
    try:
        import torch
        print(f"✓ torch 导入成功 (版本: {torch.__version__})")
        print(f"  CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA 设备数量: {torch.cuda.device_count()}")
            print(f"  当前 CUDA 设备: {torch.cuda.current_device()}")
    except Exception as e:
        print(f"✗ torch 导入失败!")
        print(f"  错误类型: {type(e).__name__}")
        print(f"  错误: {e}")
        print(f"  代码行: test_2d_keypoints.py, line 23")
        traceback.print_exc()
        return False
    
    print("[DEBUG] 开始测试 cv2 导入...")
    try:
        import cv2
        print(f"✓ opencv-python 导入成功 (版本: {cv2.__version__})")
    except Exception as e:
        print(f"✗ opencv-python 导入失败!")
        print(f"  错误类型: {type(e).__name__}")
        print(f"  错误: {e}")
        print(f"  代码行: test_2d_keypoints.py, line 35")
        traceback.print_exc()
        return False
    
    print("[DEBUG] 开始测试 numpy 导入...")
    try:
        import numpy as np
        print(f"✓ numpy 导入成功 (版本: {np.__version__})")
    except Exception as e:
        print(f"✗ numpy 导入失败!")
        print(f"  错误类型: {type(e).__name__}")
        print(f"  错误: {e}")
        print(f"  代码行: test_2d_keypoints.py, line 43")
        traceback.print_exc()
        return False
    
    print()
    return True

def test_model_manager_import():
    """测试 ModelManager 导入"""
    print("=" * 60)
    print("第2步: 测试 ModelManager 导入")
    print("=" * 60)
    
    print("[DEBUG] 开始导入 ModelManager...")
    print(f"[DEBUG] 当前目录: {current_dir}")
    print(f"[DEBUG] ModelManager 路径: {os.path.join(current_dir, 'PPCAlib', 'model_manager.py')}")
    
    try:
        from PPCAlib.model_manager import ModelManager
        print("✓ ModelManager 导入成功")
        print(f"[DEBUG] ModelManager 类型: {type(ModelManager)}")
        print()
        return True
    except Exception as e:
        print(f"✗ ModelManager 导入失败!")
        print(f"  错误类型: {type(e).__name__}")
        print(f"  错误: {e}")
        print(f"  文件位置: {os.path.join(current_dir, 'PPCAlib', 'model_manager.py')}")
        print(f"  代码行: test_2d_keypoints.py, line 58")
        traceback.print_exc()
        print()
        return False

def test_pose_estimator_import():
    """测试 PoseEstimator 导入"""
    print("=" * 60)
    print("第3步: 测试 PoseEstimator 导入")
    print("=" * 60)
    
    print("[DEBUG] 开始导入 PoseEstimator...")
    print(f"[DEBUG] PoseEstimator 路径: {os.path.join(current_dir, 'PPCAlib', 'pose_estimator.py')}")
    
    try:
        from PPCAlib.pose_estimator import PoseEstimator
        print("✓ PoseEstimator 导入成功")
        print(f"[DEBUG] PoseEstimator 类型: {type(PoseEstimator)}")
        print()
        return True
    except Exception as e:
        print(f"✗ PoseEstimator 导入失败!")
        print(f"  错误类型: {type(e).__name__}")
        print(f"  错误: {e}")
        print(f"  文件位置: {os.path.join(current_dir, 'PPCAlib', 'pose_estimator.py')}")
        print(f"  代码行: test_2d_keypoints.py, line 76")
        traceback.print_exc()
        print()
        return False

def test_model_loading():
    """测试模型加载"""
    print("=" * 60)
    print("第4步: 测试模型加载")
    print("=" * 60)
    
    try:
        from PPCAlib.model_manager import ModelManager
        
        print("[DEBUG] 开始初始化 ModelManager...")
        model_manager = ModelManager()
        print("✓ ModelManager 初始化成功")
        print(f"[DEBUG] device: {model_manager.device}")
        
        print("\n[DEBUG] 开始加载 HRNet 模型...")
        print("[DEBUG] 调用 model_manager.load_hrnet_model()...")
        model_manager.load_hrnet_model()
        print("✓ HRNet 模型加载成功")
        
        # 检查模型属性
        print("\n[DEBUG] 检查模型属性:")
        print(f"  hrnet_model: {type(model_manager.hrnet_model)}")
        print(f"  hrnet_model is None: {model_manager.hrnet_model is None}")
        print(f"  human_model: {type(model_manager.human_model)}")
        print(f"  human_model is None: {model_manager.human_model is None}")
        print(f"  people_sort: {type(model_manager.people_sort)}")
        print(f"  people_sort is None: {model_manager.people_sort is None}")
        print(f"  hrnet_args.det_dim: {model_manager.hrnet_args.det_dim}")
        print(f"  hrnet_args.thred_score: {model_manager.hrnet_args.thred_score}")
        
        print()
        return model_manager
        
    except FileNotFoundError as e:
        print(f"✗ 模型文件未找到!")
        print(f"  错误类型: {type(e).__name__}")
        print(f"  错误: {e}")
        print(f"  代码行: model_manager.py, load_hrnet_model() 方法")
        print(f"  请检查以下路径是否存在:")
        print(f"    - PoseFormerV2/demo/lib/hrnet/experiments/w48_384x288_adam_lr1e-3.yaml")
        print(f"    - PoseFormerV2/demo/lib/checkpoint/pose_hrnet_w48_384x288.pth")
        traceback.print_exc()
        print()
        return None
        
    except Exception as e:
        print(f"✗ 模型加载失败!")
        print(f"  错误类型: {type(e).__name__}")
        print(f"  错误: {e}")
        print(f"  代码行: test_2d_keypoints.py, line 106-108")
        traceback.print_exc()
        print()
        return None

def test_pose_estimator_init(model_manager):
    """测试 PoseEstimator 初始化"""
    print("=" * 60)
    print("第5步: 测试 PoseEstimator 初始化")
    print("=" * 60)
    
    print("[DEBUG] 开始初始化 PoseEstimator...")
    
    try:
        from PPCAlib.pose_estimator import PoseEstimator
        
        print("[DEBUG] 调用 PoseEstimator(model_manager)...")
        pose_estimator = PoseEstimator(model_manager)
        print("✓ PoseEstimator 初始化成功")
        print(f"[DEBUG] pose_estimator 类型: {type(pose_estimator)}")
        print(f"  joints_left: {pose_estimator.joints_left}")
        print(f"  joints_right: {pose_estimator.joints_right}")
        print(f"[DEBUG] model_manager 引用正常: {pose_estimator.model_manager is not None}")
        print()
        return pose_estimator
        
    except Exception as e:
        print(f"✗ PoseEstimator 初始化失败!")
        print(f"  错误类型: {type(e).__name__}")
        print(f"  错误: {e}")
        print(f"  代码行: pose_estimator.py, line 31 (__init__ 方法)")
        traceback.print_exc()
        print()
        return None

def create_test_image():
    """创建一个测试图像"""
    print("=" * 60)
    print("第6步: 创建测试图像")
    print("=" * 60)
    
    print("[DEBUG] 开始创建测试图像...")
    
    try:
        import numpy as np
        import cv2
        
        print("[DEBUG] 创建 640x480 空白图像...")
        # 创建一个简单的测试图像 (640x480, RGB)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        print("[DEBUG] 绘制简单人形...")
        # 绘制一个简单的人形（用于测试）
        # 头部
        cv2.circle(frame, (320, 100), 30, (255, 255, 255), -1)
        # 身体
        cv2.rectangle(frame, (280, 130), (360, 280), (255, 255, 255), -1)
        # 左臂
        cv2.rectangle(frame, (220, 130), (280, 250), (255, 255, 255), -1)
        # 右臂
        cv2.rectangle(frame, (360, 130), (420, 250), (255, 255, 255), -1)
        # 左腿
        cv2.rectangle(frame, (280, 280), (310, 420), (255, 255, 255), -1)
        # 右腿
        cv2.rectangle(frame, (330, 280), (360, 420), (255, 255, 255), -1)
        
        print("✓ 测试图像创建成功")
        print(f"  图像尺寸: {frame.shape}")
        print(f"  图像数据类型: {frame.dtype}")
        print()
        return frame
        
    except Exception as e:
        print(f"✗ 测试图像创建失败!")
        print(f"  错误类型: {type(e).__name__}")
        print(f"  错误: {e}")
        print(f"  代码行: test_2d_keypoints.py, line 177-191")
        traceback.print_exc()
        print()
        return None

def test_get_2d_keypoints_with_realsense(pose_estimator):
    """使用RealSense相机测试 get_2d_keypoints 方法"""
    print("=" * 60)
    print("第7步(备选): 使用RealSense相机测试 get_2d_keypoints 方法")
    print("=" * 60)
    
    print("[DEBUG] 尝试初始化RealSense相机...")
    
    try:
        import pyrealsense2 as rs
        import numpy as np
        
        # 配置RealSense相机
        print("[DEBUG] 配置RealSense pipeline...")
        pipeline = rs.pipeline()
        config = rs.config()
        
        # 启用彩色流
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        print("[DEBUG] 启动RealSense相机...")
        pipeline.start(config)
        
        print("✓ RealSense相机启动成功")
        print("[DEBUG] 等待相机稳定...")
        
        # 等待相机稳定，跳过前几帧
        for i in range(30):
            pipeline.wait_for_frames()
        
        print("[DEBUG] 获取一帧图像...")
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        
        if not color_frame:
            print("✗ 未能获取彩色帧")
            pipeline.stop()
            return False
        
        # 转换为numpy数组
        frame = np.asanyarray(color_frame.get_data())
        print(f"✓ 获取RealSense图像成功")
        print(f"[DEBUG] frame 形状: {frame.shape}")
        print(f"[DEBUG] frame 类型: {type(frame)}")
        print(f"[DEBUG] frame 数据类型: {frame.dtype}")
        
        # 测试get_2d_keypoints
        print("\n[DEBUG] ===== 开始执行 get_2d_keypoints (RealSense输入) =====")
        print("[DEBUG] 调用 pose_estimator.get_2d_keypoints(frame)...")
        print("[DEBUG] 对应代码位置: pose_estimator.py, line 38")
        
        print("\n[DEBUG] 第1步: YOLO 人体检测 (yolo_det)...")
        print("[DEBUG] 对应代码: pose_estimator.py, line 40-42")
        input_2D_no, input_2D_sc = pose_estimator.get_2d_keypoints(frame)
        print("[DEBUG] YOLO 检测完成")
        
        print("\n✓ get_2d_keypoints (RealSense输入) 执行成功!")
        print("[DEBUG] ===== get_2d_keypoints 执行完毕 =====\n")
        
        print(f"返回结果:")
        print(f"  input_2D_no 形状: {input_2D_no.shape}")
        print(f"  input_2D_no 类型: {type(input_2D_no)}")
        print(f"  input_2D_no 数据类型: {input_2D_no.dtype}")
        print(f"  input_2D_sc 形状: {input_2D_sc.shape}")
        print(f"  input_2D_sc 类型: {type(input_2D_sc)}")
        print(f"  input_2D_sc 数据类型: {input_2D_sc.dtype}")
        
        print(f"\n关键点坐标示例 (前5个):")
        for i in range(min(5, len(input_2D_no))):
            print(f"  关键点 {i}: 位置=[{input_2D_no[i][0]:.1f}, {input_2D_no[i][1]:.1f}], 置信度={input_2D_sc[i][0]:.3f}")
        
        # 关闭相机
        print("\n[DEBUG] 关闭RealSense相机...")
        pipeline.stop()
        print("✓ RealSense相机已关闭")
        print()
        return True
        
    except RuntimeError as e:
        print(f"\n✗ RealSense相机错误!")
        print(f"  错误类型: {type(e).__name__}")
        print(f"  错误: {e}")
        print(f"  可能原因:")
        print(f"    - RealSense相机未连接")
        print(f"    - RealSense相机被其他程序占用")
        print(f"    - RealSense驱动未安装")
        print(f"\n完整的错误堆栈:")
        traceback.print_exc()
        print()
        return False
        
    except ImportError as e:
        print(f"\n✗ pyrealsense2 导入失败!")
        print(f"  错误类型: {type(e).__name__}")
        print(f"  错误: {e}")
        print(f"  请安装: pip install pyrealsense2")
        print(f"\n完整的错误堆栈:")
        traceback.print_exc()
        print()
        return False
        
    except Exception as e:
        print(f"\n✗ RealSense测试失败!")
        print(f"  错误类型: {type(e).__name__}")
        print(f"  错误信息: {e}")
        print(f"\n详细错误堆栈:")
        traceback.print_exc()
        print()
        
        # 尝试关闭相机
        try:
            pipeline.stop()
        except:
            pass
        
        return False

def test_get_2d_keypoints(pose_estimator, frame):
    """测试 get_2d_keypoints 方法"""
    print("=" * 60)
    print("第7步: 测试 get_2d_keypoints 方法 (合成图像)")
    print("=" * 60)
    
    print("[DEBUG] 准备调用 get_2d_keypoints...")
    print(f"[DEBUG] frame 形状: {frame.shape}")
    print(f"[DEBUG] frame 类型: {type(frame)}")
    
    try:
        print("\n[DEBUG] ===== 开始执行 get_2d_keypoints =====")
        print("[DEBUG] 调用 pose_estimator.get_2d_keypoints(frame)...")
        print("[DEBUG] 对应代码位置: pose_estimator.py, line 38")
        
        # 这是关键的测试行
        print("\n[DEBUG] 第1步: YOLO 人体检测 (yolo_det)...")
        print("[DEBUG] 对应代码: pose_estimator.py, line 40-42")
        input_2D_no, input_2D_sc = pose_estimator.get_2d_keypoints(frame)
        print("[DEBUG] YOLO 检测完成")
        
        print("\n✓ get_2d_keypoints 执行成功!")
        print("[DEBUG] ===== get_2d_keypoints 执行完毕 =====\n")
        
        print(f"返回结果:")
        print(f"  input_2D_no 形状: {input_2D_no.shape}")
        print(f"  input_2D_no 类型: {type(input_2D_no)}")
        print(f"  input_2D_no 数据类型: {input_2D_no.dtype}")
        print(f"  input_2D_sc 形状: {input_2D_sc.shape}")
        print(f"  input_2D_sc 类型: {type(input_2D_sc)}")
        print(f"  input_2D_sc 数据类型: {input_2D_sc.dtype}")
        
        print(f"\n关键点坐标示例 (前5个):")
        for i in range(min(5, len(input_2D_no))):
            print(f"  关键点 {i}: 位置=[{input_2D_no[i][0]:.1f}, {input_2D_no[i][1]:.1f}], 置信度={input_2D_sc[i][0]:.3f}")
        
        print()
        return True
        
    except AttributeError as e:
        print(f"\n✗ 属性错误!")
        print(f"  错误类型: {type(e).__name__}")
        print(f"  错误: {e}")
        print(f"  可能原因: model_manager 缺少必要的属性")
        print(f"  检查 model_manager 是否正确初始化了以下属性:")
        print(f"    - human_model")
        print(f"    - hrnet_model")
        print(f"    - people_sort")
        print(f"    - hrnet_args.det_dim")
        print(f"    - hrnet_args.thred_score")
        print(f"\n完整的错误堆栈:")
        traceback.print_exc()
        print()
        return False
        
    except RuntimeError as e:
        print(f"\n✗ 运行时错误!")
        print(f"  错误类型: {type(e).__name__}")
        print(f"  错误: {e}")
        print(f"  可能原因:")
        print(f"    - CUDA 内存不足")
        print(f"    - 模型权重不匹配")
        print(f"    - 张量设备不一致 (CPU/GPU)")
        print(f"    - 输入数据格式不正确")
        print(f"\n完整的错误堆栈:")
        traceback.print_exc()
        print()
        return False
        
    except ImportError as e:
        print(f"\n✗ 导入错误!")
        print(f"  错误类型: {type(e).__name__}")
        print(f"  错误: {e}")
        print(f"  可能原因: PoseFormerV2 相关模块未找到")
        print(f"\n完整的错误堆栈:")
        traceback.print_exc()
        print()
        return False
        
    except Exception as e:
        print(f"\n✗ get_2d_keypoints 执行失败!")
        print(f"  错误类型: {type(e).__name__}")
        print(f"  错误信息: {e}")
        print(f"\n详细错误堆栈 (定位具体代码行):")
        traceback.print_exc()
        print()
        return False

def main():
    """主测试流程"""
    print("\n" + "=" * 60)
    print("开始测试 pose_estimator.get_2d_keypoints()")
    print("=" * 60)
    print(f"[DEBUG] Python 版本: {sys.version}")
    print(f"[DEBUG] 当前工作目录: {os.getcwd()}")
    print(f"[DEBUG] 脚本目录: {current_dir}")
    print()
    
    # 第1步: 测试基础导入
    if not test_imports():
        print("❌ 测试失败: 基础库导入失败")
        return
    
    # 第2步: 测试 ModelManager 导入
    if not test_model_manager_import():
        print("❌ 测试失败: ModelManager 导入失败")
        return
    
    # 第3步: 测试 PoseEstimator 导入
    if not test_pose_estimator_import():
        print("❌ 测试失败: PoseEstimator 导入失败")
        return
    
    # 第4步: 测试模型加载
    model_manager = test_model_loading()
    if model_manager is None:
        print("❌ 测试失败: 模型加载失败")
        return
    
    # 第5步: 测试 PoseEstimator 初始化
    pose_estimator = test_pose_estimator_init(model_manager)
    if pose_estimator is None:
        print("❌ 测试失败: PoseEstimator 初始化失败")
        return
    
    # 第6步: 创建测试图像
    frame = create_test_image()
    if frame is None:
        print("❌ 测试失败: 测试图像创建失败")
        return
    
    # 第7步: 测试 get_2d_keypoints 方法（合成图像）
    if not test_get_2d_keypoints(pose_estimator, frame):
        print("❌ 测试失败: get_2d_keypoints 方法执行失败")
        return
    
    # 第8步: 测试 get_2d_keypoints 方法（RealSense相机）
    print("=" * 60)
    print("第8步: 使用RealSense相机测试 (可选)")
    print("=" * 60)
    print("提示: 如果没有RealSense相机，此测试将被跳过")
    user_input = input("是否测试RealSense相机? (y/n，默认n): ").strip().lower()
    
    if user_input == 'y' or user_input == 'yes':
        print("\n开始RealSense相机测试...")
        if test_get_2d_keypoints_with_realsense(pose_estimator):
            print("✓ RealSense相机测试通过")
        else:
            print("⚠ RealSense相机测试失败（但不影响整体测试结果）")
    else:
        print("跳过RealSense相机测试")
    
    print()
    
    # 所有测试通过
    print("=" * 60)
    print("✅✅✅ 所有必要测试通过! ✅✅✅")
    print("=" * 60)
    print()
    print("[总结] get_2d_keypoints() 方法测试完全通过!")
    print()
    print("如果此测试在本机通过但在其他电脑失败，请检查:")
    print("1. Python 版本是否一致")
    print("2. 依赖库版本是否一致 (torch, opencv-python, numpy)")
    print("3. 模型文件是否完整存在")
    print("   - PoseFormerV2/demo/lib/hrnet/experiments/w48_384x288_adam_lr1e-3.yaml")
    print("   - PoseFormerV2/demo/lib/checkpoint/pose_hrnet_w48_384x288.pth")
    print("4. CUDA 版本是否兼容 (如果使用GPU)")
    print("5. 系统权限是否足够")
    print("6. PoseFormerV2 文件夹是否完整复制")
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
    except Exception as e:
        print("\n\n" + "=" * 60)
        print("❌ 测试过程中发生未捕获的异常!")
        print("=" * 60)
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {e}")
        print("\n完整的错误堆栈:")
        traceback.print_exc()
