import numpy as np
import pyrealsense2 as rs


class CameraManager:
    """相机管理器 - 仅支持RGBD相机"""

    def __init__(self):
        self.pipeline = None
        self.align = None
        self.pipeline_started = False

    def initialize(self):
        """初始化RGBD相机"""
        if self.pipeline is not None:
            return

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

        try:
            self.pipeline.start(config)
            self.pipeline_started = True
            self.align = rs.align(rs.stream.color)
            print("RGBD camera initialized")
        except RuntimeError as exc:
            self.release()
            raise RuntimeError(f"Failed to start RGBD camera: {exc}") from exc

    def read_rgbd_frame(self):
        """读取RGBD帧"""
        if self.pipeline is None:
            raise RuntimeError("Camera pipeline not initialized")

        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            return False, None, None, None

        frame = np.asanyarray(color_frame.get_data())
        depth_intrinsics = depth_frame.profile.as_video_stream_profile().get_intrinsics()

        return True, frame, depth_frame, depth_intrinsics

    def robot_camera_coord(self):
        """返回机器人相机坐标偏移"""
        return np.array([0.126, 1.126, 0.317])

    def release(self):
        """释放相机资源"""
        if self.pipeline and self.pipeline_started:
            try:
                self.pipeline.stop()
            except RuntimeError:
                pass
        self.pipeline = None
        self.align = None
        self.pipeline_started = False
