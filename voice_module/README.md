# PPCA2 实时姿态估计与碰撞检测

本项目实现基于 YOLO 姿态估计与几何碰撞检测的人机协作安全演示，可接入 Intel RealSense 相机与（可选）真实机械臂/仿真环境。

> 运行平台：Windows 10/11 + PowerShell + Python 3.9（Conda 环境：HMP2）

---

## 环境准备

- 建议使用 Conda（已知工作环境：`HMP2`，Python 3.9）
- NVIDIA GPU 可选；若使用 GPU，请确保显卡驱动与 CUDA 对应版本（示例：CUDA 12.6）
- RealSense 相机（可选），安装 Intel RealSense SDK（建议使用 Intel RealSense Viewer 确认设备正常）

### 1) 创建并激活 Conda 环境（如本机未已有 HMP2）

```powershell
conda create -n HMP2 python=3.9 -y
conda activate HMP2
```

### 2) 安装 PyTorch（请先按你的硬件选择）

- CUDA 12.6（GPU）：

```powershell
pip install torch==2.8.0+cu126 torchvision==0.23.0+cu126 --index-url https://download.pytorch.org/whl/cu126
```

- 仅 CPU：

```powershell
pip install torch==2.8.0+cpu torchvision==0.23.0+cpu --index-url https://download.pytorch.org/whl/cpu
```

> 如需其它 CUDA 版本，请到 https://pytorch.org/ 生成官方安装命令。

### 3) 安装项目依赖

在项目根目录执行：

```powershell
pip install -r requirements.txt
```

- 若 `python-fcl` 在 Windows 下通过 pip 安装失败，可用 Conda Forge：

```powershell
conda install -c conda-forge python-fcl
```

- 如需 RealSense：确保 `pyrealsense2` 安装成功；若相机启动失败，参考下文“常见问题”。

---

## 项目结构（节选）

- `checkpoint/yolo11n-pose.pt`：YOLO 姿态估计权重（已包含）。
- `config/realtime_system.yml`：系统参数配置文件。
- `PPCAlib/`：关键库，包含传感器、模型、碰撞检测、可视化等模块。
- `RealManmodel/rm_75_b_description.urdf`：机械臂 URDF 模型。
- `src/`：主程序入口（含不同配置的启动脚本）。
- `test/main_detect.py`：最小化 RealSense + YOLO 姿态检测示例。

---

## 运行方式

确保已在 PowerShell 中激活 Conda 环境 `HMP2`：

```powershell
conda activate HMP2
```

### 1) 仅测试相机 + 姿态检测

```powershell
python .\test\main_detect.py
```

- 默认读取 `checkpoint/yolo11n-pose.pt` 权重。
- 窗口中按 `q` 退出。

### 2) 启动完整系统（读取配置文件）

```powershell
python .\src\main.py
```

- 配置文件：`config/realtime_system.yml`
- 支持的配置键（示例）：
  - `realtime_system.robot_base_position`: 机械臂基座/立方体中心位置 `[x, y, z]`
  - `realtime_system.capsule_radius`: 胶囊体半径（人体关键点碰撞近似）
  - `realtime_system.cube_size`: 场景立方体尺寸
  - `realtime_system.enable_prediction`: 是否启用人体运动预测
  - `realtime_system.realrobot`: 是否连接真实机械臂（否则为仿真/可视化）

也可使用不带预测的版本：

```powershell
python .\src\main_nopred.py
```

---

## 常见问题与排查

### A. RealSense 相机启动失败
- 安装并用 Intel RealSense Viewer 验证设备与固件。
- 使用 USB 3.x 高速口，避免扩展坞影响。
- 确保 `pyrealsense2` 匹配你的 SDK/固件版本，优先使用和本仓库相同的 `2.50.0.3812`。

### B. PyTorch 安装/运行报错（CUDA 相关）
- 版本需和显卡驱动、CUDA 工具链匹配；务必按 https://pytorch.org/ 的安装指令执行。
- 如果仅 CPU 推理，使用 CPU 版本命令即可。

### C. python-fcl 安装问题（Windows）
- 首选 Conda Forge 渠道：`conda install -c conda-forge python-fcl`
- 若仍失败，请确认已安装 VS 运行库（Microsoft C++ 运行时）并重试。

### D. OpenCV 显示窗口问题
- 在远程/无桌面环境下 `cv2.imshow` 可能无法创建窗口，请本地运行或改为保存图片调试。

---

## 参考与致谢
- YOLO/Ultralytics: https://github.com/ultralytics/ultralytics
- Intel RealSense SDK: https://github.com/IntelRealSense/librealsense
- PyTorch: https://pytorch.org/

---

## 许可证
仅供研究与教学用途。若用于商业目的，请自行评估并遵循各第三方依赖的许可证条款。
