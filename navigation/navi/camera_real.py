import cv2
import pyrealsense2 as rs
from ultralytics import YOLO
import numpy as np
import time


# 一些统一的绘制参数，方便统一修改
BOX_COLOR = (0, 140, 255)       # 检测框颜色 (BGR): 橙色
BOX_THICKNESS = 2
KPT_COLOR = (0, 255, 0)         # 关键点颜色: 绿色
KPT_RADIUS = 3
SKELETON_COLOR = (0, 255, 255)  # 骨架颜色: 青色
SKELETON_THICKNESS = 2
TEXT_COLOR = (255, 255, 255)    # 文本颜色: 白色
TEXT_BG_COLOR = (0, 0, 0)       # 文本背景颜色: 黑色
TEXT_SCALE = 0.6
TEXT_THICKNESS = 1


def draw_text_with_background(img, text, org,
                              font=cv2.FONT_HERSHEY_SIMPLEX,
                              font_scale=TEXT_SCALE,
                              text_color=TEXT_COLOR,
                              bg_color=TEXT_BG_COLOR,
                              thickness=TEXT_THICKNESS,
                              alpha=0.6,
                              padding=3):
    """
    在图像上绘制带背景的文字（背景半透明）。
    org: 左下角坐标 (x, y)
    """
    # 计算文字尺寸
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org
    # 背景矩形左上角和右下角
    x1, y1 = x - padding, y - th - baseline - padding
    x2, y2 = x + tw + padding, y + baseline + padding

    # 防止越界
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, img.shape[1] - 1)
    y2 = min(y2, img.shape[0] - 1)

    # 复制一层用于做半透明混合
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_color, -1)
    # 叠加
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # 写文字
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)


def draw_fancy_box(img, x1, y1, x2, y2, color=BOX_COLOR, thickness=2, corner_len=15):
    """
    画一个带角标的矩形框，比普通 rectangle 更有视觉效果
    """
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

    # 四个角加装饰线段
    # 左上角
    cv2.line(img, (x1, y1), (x1 + corner_len, y1), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y1), (x1, y1 + corner_len), color, thickness, cv2.LINE_AA)
    # 右上角
    cv2.line(img, (x2, y1), (x2 - corner_len, y1), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2, y1 + corner_len), color, thickness, cv2.LINE_AA)
    # 左下角
    cv2.line(img, (x1, y2), (x1 + corner_len, y2), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x1, y2 - corner_len), color, thickness, cv2.LINE_AA)
    # 右下角
    cv2.line(img, (x2, y2), (x2 - corner_len, y2), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y2), (x2, y2 - corner_len), color, thickness, cv2.LINE_AA)


def draw_skeleton(image, keypoints, conf_threshold=0.2):
    """
    在图像上画人体关键点和骨架连接线。
    keypoints: (num_kpts, 3) -> [x, y, conf]
    """
    # COCO 标准骨架连接（根据 YOLO pose 的定义）
    skeleton = [
        (0, 1), (0, 2), (1, 3), (2, 4),
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12), (11, 13),
        (13, 15), (12, 14), (14, 16)
    ]

    # 可选：不同部分不同颜色（例如躯干、手臂、腿）
    # 这里简单都用同一种颜色 SKELETON_COLOR，你也可以改成一个颜色列表

    # 画骨架线（先画线再画点，视觉更好）
    for (i, j) in skeleton:
        if i < len(keypoints) and j < len(keypoints):
            x1, y1, c1 = keypoints[i]
            x2, y2, c2 = keypoints[j]
            if c1 > conf_threshold and c2 > conf_threshold:
                cv2.line(
                    image,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    SKELETON_COLOR,
                    SKELETON_THICKNESS,
                    cv2.LINE_AA,
                )

    # 画关键点
    for idx, (x, y, c) in enumerate(keypoints):
        if c < conf_threshold:
            continue
        cv2.circle(image, (int(x), int(y)), KPT_RADIUS, KPT_COLOR, -1, cv2.LINE_AA)


def main():
    # 1. 加载 YOLOv11 pose 模型
    model_path = "yolo11n-pose.pt"   # 改成你的模型路径
    model = YOLO(model_path)
    # model.to("cuda")  # 如果有 GPU，可以打开

    # 2. 配置 RealSense D435 管线
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # 如需深度可打开：
    # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    profile = pipeline.start(config)

    # 对齐到彩色
    align_to = rs.stream.color
    align = rs.align(align_to)

    # 创建可调整大小的窗口，并且不用 Qt
    win_name = "Surgeon Pose"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
    # 你也可以预设一个起始大小（之后你仍然可以手动调整）
    cv2.resizeWindow(win_name, 960, 540)

    try:
        while True:
            # 3. 读取 RealSense 帧
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)

            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())

            # 4. YOLOv11 推理
            t0 = time.time()
            results = model(color_image, verbose=False)[0]
            infer_time = (time.time() - t0) * 1000.0  # ms

            annotated_img = color_image.copy()

            if results.keypoints is not None and len(results.keypoints) > 0:
                kpts = results.keypoints.data.cpu().numpy()  # (N, K, 3)
                boxes = results.boxes

                for i, person_keypoints in enumerate(kpts):
                    box = boxes[i]
                    cls_id = int(box.cls[0].item())
                    cls_name = results.names.get(cls_id, str(cls_id))
                    conf = float(box.conf[0].item())

                    # 只画 person
                    if cls_name.lower() != "person":
                        continue

                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    # 画更好看的带角框
                    # draw_fancy_box(annotated_img, x1, y1, x2, y2,
                    #                color=BOX_COLOR, thickness=BOX_THICKNESS)

                    # 标签放在框的左上角附近
                    # label = f"{cls_name} {conf:.2f}"
                    # draw_text_with_background(
                    #     annotated_img,
                    #     label,
                    #     (x1 + 3, max(20, y1 - 5)),
                    #     font_scale=0.6,
                    #     text_color=(255, 255, 255),
                    #     bg_color=(0, 0, 0),
                    #     alpha=0.6
                    # )

                    # 画关键点和骨架
                    draw_skeleton(annotated_img, person_keypoints, conf_threshold=0.2)

            # # 左上角显示 FPS / 推理时间
            # draw_text_with_background(
            #     annotated_img,
            #     f"Infer: {infer_time:.1f} ms",
            #     (10, 30),
            #     font_scale=0.7,
            #     bg_color=(0, 0, 0),
            #     alpha=0.4,
            # )

            # 5. 显示图像（窗口大小你可以自己调）
            cv2.imshow(win_name, annotated_img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
