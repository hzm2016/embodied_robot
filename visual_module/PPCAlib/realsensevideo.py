import cv2
import numpy as np
import pyrealsense2 as rs
from typing import Any, Optional, Tuple


def find_l515(verbose: bool = True) -> bool:
    devices = rs.context().query_devices()
    if len(devices) == 0:
        if verbose:
            print("未找到任何RealSense设备")
        return False

    found = False
    for dev in devices:
        name = dev.get_info(rs.camera_info.name)
        if verbose:
            print(f"找到设备：{name}")
        if "L515" in name.upper():
            found = True
    if verbose and found:
        print("✅ 成功识别L515相机！")
    if verbose and not found:
        print("⚠️ 已连接RealSense但未检测到L515型号")
    return found


class _MouseState:
    def __init__(self) -> None:
        self.depth_frame: Optional[rs.depth_frame] = None
        self.depth_intrinsics: Optional[Any] = None
        self.last_click: Optional[Tuple[int, int, float, Tuple[float, float, float]]] = None

    def on_mouse(self, event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and self.depth_frame is not None and self.depth_intrinsics is not None:
            depth = self.depth_frame.get_distance(x, y)
            if depth > 0:
                point = tuple(rs.rs2_deproject_pixel_to_point(self.depth_intrinsics, [x, y], depth))
                print(
                    f"像素({x}, {y}) 深度: {depth:.3f} m -> 3D坐标: "
                    f"X={point[0]:.3f} m, Y={point[1]:.3f} m, Z={point[2]:.3f} m"
                )
            else:
                print(f"像素({x}, {y}) 深度无效")
                point = (0.0, 0.0, 0.0)
            self.last_click = (x, y, depth, point)


def stream_l515(color_resolution: Tuple[int, int] = (1280, 720),
                depth_resolution: Tuple[int, int] = (640, 480),
                fps: int = 30) -> None:
    if not find_l515(verbose=False):
        print("未检测到L515，请检查连接后重试。")
        return

    state = _MouseState()
    align = rs.align(rs.stream.color)
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, depth_resolution[0], depth_resolution[1], rs.format.z16, fps)
    config.enable_stream(rs.stream.color, color_resolution[0], color_resolution[1], rs.format.bgr8, fps)

    cv2.namedWindow("L515 Color", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("L515 Depth", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("L515 Color", state.on_mouse)

    print("按 q 退出，鼠标左键点击彩色图像获取深度值。")

    try:
        pipeline.start(config)
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            state.depth_frame = depth_frame
            state.depth_intrinsics = depth_frame.profile.as_video_stream_profile().get_intrinsics()

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.05), cv2.COLORMAP_JET)

            if state.last_click is not None:
                x, y, depth, point = state.last_click
                label = f"{depth:.3f} m" if depth > 0 else "无效深度"
                cv2.circle(color_image, (x, y), 4, (0, 0, 255), -1)
                cv2.putText(color_image, label, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                if depth > 0:
                    coord_label = f"X:{point[0]:.3f} Y:{point[1]:.3f} Z:{point[2]:.3f} m"
                    cv2.putText(color_image, coord_label, (x + 10, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            cv2.imshow("L515 Color", color_image)
            cv2.imshow("L515 Depth", depth_colormap)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    except Exception as exc:
        print(f"流式传输失败：{exc}")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    stream_l515()