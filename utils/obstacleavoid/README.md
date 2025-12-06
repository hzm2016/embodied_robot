# 实时姿态估计与碰撞检测系统

## 项目结构

重构后的项目采用模块化设计，每个类分离到独立的文件中，提高代码的可维护性和可读性。

### 文件结构

```
e:\Code\PPCA\
├── main.py                          # 主程序入口文件
├── realtime_system.py               # 主控制器系统类
├── joint_mapper.py                  # 关节映射器
├── human_collision_detector.py      # 人体碰撞检测器
├── model_manager.py                 # AI模型管理器
├── pose_estimator.py               # 姿态估计器
├── collision_visualizer.py         # 碰撞可视化器
├── camera_manager.py               # 摄像头管理器
├── Kinematics.py                   # 机械臂运动学
└── realtime_pose_collision_detection_visual.py  # 原始单体文件（已废弃）
```

### 模块说明

#### 1. `main.py` - 主程序入口
- 程序启动点
- 系统参数配置
- 异常处理和资源清理

#### 2. `realtime_system.py` - 主控制器
- `RealtimePoseCollisionSystem` 类
- 集成所有组件的主控制器
- 处理主循环逻辑和用户交互

#### 3. `joint_mapper.py` - 关节映射器
- `JointMapper` 类
- 将17关节姿态映射到10关节
- 支持PoseFormer到碰撞检测的数据转换

#### 4. `human_collision_detector.py` - 人体碰撞检测器
- `HumanCollisionDetector` 类
- FCL碰撞检测库集成
- 人体胶囊与立方体碰撞计算

#### 5. `model_manager.py` - AI模型管理器
- `ModelManager` 类
- PoseFormerV2模型加载
- HRNet模型加载
- YOLO和SORT模型管理

#### 6. `pose_estimator.py` - 姿态估计器
- `PoseEstimator` 类
- 2D关键点检测
- 3D姿态估计
- 数据预处理和后处理

#### 7. `collision_visualizer.py` - 碰撞可视化器
- `CollisionVisualizer` 类
- 2D姿态绘制
- 3D碰撞场景可视化
- matplotlib交互式3D显示

#### 8. `camera_manager.py` - 摄像头管理器
- `CameraManager` 类
- 摄像头初始化和配置
- 帧读取和资源管理

## 使用方法

### 运行程序
```bash
python main.py
```

### 控制说明
- **'q'** - 退出程序
- **'r'** - 重置立方体位置
- **'c'** - 改变立方体大小
- **空格键** - 暂停/恢复
- **鼠标** - 在3D窗口中拖拽旋转视角

## 优势

### 1. 模块化设计
- 每个类职责单一，便于理解和维护
- 模块间低耦合，高内聚
- 便于单元测试和调试

### 2. 代码复用性
- 模块可以独立使用
- 便于在其他项目中复用组件
- 支持功能扩展

### 3. 维护性
- 修改一个功能只需关注对应模块
- 降低代码复杂度
- 便于团队协作开发

### 4. 扩展性
- 易于添加新的碰撞检测算法
- 支持不同的可视化方式
- 可以轻松集成新的AI模型

## 技术栈

- **深度学习**: PoseFormerV2, HRNet, YOLO
- **碰撞检测**: FCL (Flexible Collision Library)
- **计算机视觉**: OpenCV
- **3D可视化**: matplotlib
- **数学计算**: NumPy, PyTorch
- **目标跟踪**: SORT算法

## 依赖项

确保安装以下Python包：
- torch
- opencv-python
- matplotlib
- numpy
- fcl
- scipy

## 性能特点

- **实时性**: 30-60 FPS处理速度
- **精确性**: 17关节3D姿态估计
- **交互性**: 实时碰撞检测和可视化
- **可视化**: 双窗口显示（2D摄像头 + 3D场景）