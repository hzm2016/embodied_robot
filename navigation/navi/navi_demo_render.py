from realman_kinematics import forward_kinematics as fk
import numpy as np
import open3d as o3d
import time
import redis
import threading
import json

# -------------------------
# Global state
# -------------------------
transform_base2model = np.eye(4)  # 增量微调
base2model_nominal = np.eye(4)    # 标定值

TRANSLATION_STEP = 5.0  # mm
ROTATION_STEP = np.deg2rad(1.0)  # rad

# init_frame 在模型坐标系下的目标位姿（mm, rad）
T_model2init = None

# 机器人连接与外设
redis_client = None

REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0
JOINT_CHANNEL = "actual_joint_angle"

_joint_lock = threading.Lock()
_latest_joint_deg = None
_joint_thread = None

# 对齐选项
align_orientation = True  # True: 位置+姿态都对齐；False: 仅位置对齐

# 相机坐标系可视化长度（mm）
cam_axis_len = 40.0

# -------------------------
# Utility: Pose <-> T
# -------------------------
def pose_to_transformation_matrix(pose):
    x, y, z = pose[0:3]
    rx, ry, rz = pose[3:6]
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx),  np.cos(rx)]])
    Ry = np.array([[ np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[ np.cos(rz), -np.sin(rz), 0],
                   [ np.sin(rz),  np.cos(rz), 0],
                   [0, 0, 1]])
    R = Rz @ Ry @ Rx
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T

def transformation_matrix_to_pose(T):
    x, y, z = T[0, 3], T[1, 3], T[2, 3]
    R = T[:3, :3]
    ry = np.arctan2(R[0, 2], np.sqrt(R[0, 0]**2 + R[0, 1]**2))
    rx = np.arctan2(-R[1, 2] / (np.cos(ry) + 1e-12), R[2, 2] / (np.cos(ry) + 1e-12))
    rz = np.arctan2(-R[0, 1] / (np.cos(ry) + 1e-12), R[0, 0] / (np.cos(ry) + 1e-12))
    return np.array([x, y, z, rx, ry, rz])

# -------------------------
# Debug helpers
# -------------------------
def _print_transform_state():
    global base2model_nominal, transform_base2model
    effective = transform_base2model @ base2model_nominal
    pose_delta = transformation_matrix_to_pose(transform_base2model)
    print("transform_base2model delta pose [mm, rad]:", pose_delta)
    print("Effective base2model matrix:\n", effective)

def _apply_keyboard_transform(delta_pose):
    global transform_base2model
    delta = pose_to_transformation_matrix(delta_pose)
    transform_base2model = delta @ transform_base2model
    _print_transform_state()

def _reset_transform():
    global transform_base2model
    transform_base2model = np.eye(4)
    _print_transform_state()

def _make_delta_callback(delta_pose):
    def callback(vis):
        _apply_keyboard_transform(delta_pose)
        return False
    return callback

def _reset_callback(vis):
    _reset_transform()
    return False

def _print_callback(vis):
    _print_transform_state()
    return False

# -------------------------
# Joint subscription helpers
# -------------------------
def _joint_listener():
    global _latest_joint_deg
    client = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
    pubsub = client.pubsub()
    pubsub.subscribe(JOINT_CHANNEL)
    print(f"[JointSub] Subscribed to {JOINT_CHANNEL}")
    for message in pubsub.listen():
        if message.get("type") != "message":
            continue
        try:
            data = message.get("data")
            if isinstance(data, bytes):
                data = data.decode("utf-8")
            payload = json.loads(data)
            joint_list = payload.get("joint_angle")
            if joint_list is None:
                continue
            with _joint_lock:
                _latest_joint_deg = list(joint_list)
        except Exception as exc:
            print("[JointSub] Parse error:", exc)


def start_joint_listener():
    global _joint_thread
    if _joint_thread is None or not _joint_thread.is_alive():
        _joint_thread = threading.Thread(target=_joint_listener, daemon=True)
        _joint_thread.start()


def get_latest_joint_angles():
    with _joint_lock:
        if _latest_joint_deg is None:
            return None
        return list(_latest_joint_deg)

# -------------------------
# Geometry helpers
# -------------------------
def create_thick_line(start, end, rcm, radius=2.0, color=[1, 0, 0]):
    start = np.array(start, dtype=float)
    end = np.array(end, dtype=float)
    rcm = np.array(rcm, dtype=float)
    vector = end - start
    length = np.linalg.norm(vector) + 1e-12
    direction = vector / length

    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=float(length))
    rotation = np.eye(3)
    z_axis = np.array([0.0, 0.0, 1.0], dtype=float)
    if not np.allclose(direction, z_axis):
        axis = np.cross(z_axis, direction)
        axis_len = np.linalg.norm(axis)
        if axis_len > 1e-12:
            axis = axis / axis_len
            angle = np.arccos(np.clip(np.dot(z_axis, direction), -1.0, 1.0))
            rotation = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)

    cylinder.rotate(rotation, center=[0, 0, 0])
    cylinder.translate(start + direction * (length / 2.0))
    cylinder.paint_uniform_color(color)

    tip = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=radius * 2.0)
    tip.rotate(rotation, center=[0, 0, 0])
    tip.translate(end)
    tip.paint_uniform_color([0, 0, 1])

    rcm_ball = o3d.geometry.TriangleMesh.create_sphere(radius=radius * 1.5)
    rcm_ball.translate(rcm)
    rcm_ball.paint_uniform_color([0, 1, 0])

    return cylinder, tip, rcm_ball

def create_camera_axis_T(T_world, axis_len=40.0):
    """
    根据相机到世界的变换 T_world（R_wc, t_wc），创建坐标系几何（X=红，Y=绿，Z=蓝）
    """
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=float(axis_len), origin=[0, 0, 0])
    axis.transform(T_world)
    return axis

# -------------------------
# Endoscope kinematics (如需)
# -------------------------
def cal_endo2ee(pos):
    Ly0 = 60
    Lz0 = 70
    Lx2 = 59
    Ly2 = 102
    Lz2 = 67
    Lz43 = 53
    L = [Lz43 + pos[0], pos[1] + Lz0, pos[2] + Lz0, Lz2 - pos[3], pos[4]]
    psi12 = np.arctan2(L[1] - L[2], Ly0)
    psi34 = np.arctan2(L[3] - L[4], Lx2)
    T01 = pose_to_transformation_matrix([0, 0, L[1]/1000, 0, 0, 0])
    T12 = pose_to_transformation_matrix([0, 0, 0, 0, np.pi/2, np.pi/2 - psi12])
    T23 = pose_to_transformation_matrix([0, Ly2/1000, L[3]/1000, 0, 0, 0])
    T34 = pose_to_transformation_matrix([0, 0, L[0]/1000, 0, np.pi/2 + psi34, 0])
    T04 = T01 @ T12 @ T23 @ T34
    return T04

# -------------------------
# Confirm alignment: align endo_end to init_frame
# -------------------------
def _confirm_callback(vis):
    """
    一键确认：将当前 endo_end 与模型中的 init_frame 对齐
    endo_end 的朝向与 endobase 一致（R=I），仅位置为末端点。
    """
    global base2model_nominal, transform_base2model
    global T_model2init, endobase2ee
    global last_endo_end

    joint = get_latest_joint_angles()
    if joint is None:
        print("Confirm failed: no joint data from Redis subscriber.")
        return False

    joint_rad = joint
    T_base2ee = fk(joint_rad)

    # 1) base->endobase
    T_ee2endobase = np.linalg.inv(endobase2ee)
    T_base2endobase = T_base2ee @ T_ee2endobase

    # 2) endobase->endo_end：姿态相同（R=I），平移为末端点（在 endobase 局部坐标）
    p_end = np.array(last_endo_end, dtype=float)
    T_endobase2endoend = np.eye(4)
    T_endobase2endoend[:3, :3] = np.eye(3)
    T_endobase2endoend[:3, 3]  = p_end

    # 3) base->endo_end
    T_base2endoend = T_base2endobase @ T_endobase2endoend

    # 4) 目标：model->endo_end
    if align_orientation:
        T_model2endoend_tgt = T_model2init.copy()
    else:
        base2model_eff = transform_base2model @ base2model_nominal
        T_model2endoend_cur = base2model_eff @ T_base2endoend
        T_model2endoend_tgt = np.eye(4)
        T_model2endoend_tgt[:3, :3] = T_model2endoend_cur[:3, :3]
        T_model2endoend_tgt[:3, 3]  = T_model2init[:3, 3]

    # 5) base->model 新值
    T_base2model_new = T_model2endoend_tgt @ np.linalg.inv(T_base2endoend)
    base2model_nominal = T_base2model_new
    transform_base2model = np.eye(4)
    print("[Confirm] Aligned endo_end to init_frame with endobase-aligned orientation.")
    _print_transform_state()
    return False

# -------------------------
# Keyboard bindings
# -------------------------
def register_pose_keys(visualizer):
    def add_binding(keys, delta_pose):
        callback = _make_delta_callback(delta_pose)
        for key in keys:
            visualizer.register_key_callback(ord(key), callback)

    add_binding(("A", "a"), [-TRANSLATION_STEP, 0, 0, 0, 0, 0])
    add_binding(("D", "d"), [ TRANSLATION_STEP, 0, 0, 0, 0, 0])
    add_binding(("Q", "q"), [0,  TRANSLATION_STEP, 0, 0, 0, 0])
    add_binding(("E", "e"), [0, -TRANSLATION_STEP, 0, 0, 0, 0])
    add_binding(("W", "w"), [0, 0,  TRANSLATION_STEP, 0, 0, 0])
    add_binding(("S", "s"), [0, 0, -TRANSLATION_STEP, 0, 0, 0])

    add_binding(("I", "i"), [0, 0, 0,  ROTATION_STEP, 0, 0])
    add_binding(("K", "k"), [0, 0, 0, -ROTATION_STEP, 0, 0])
    add_binding(("J", "j"), [0, 0, 0, 0,  ROTATION_STEP, 0])
    add_binding(("L", "l"), [0, 0, 0, 0, -ROTATION_STEP, 0])
    add_binding(("U", "u"), [0, 0, 0, 0, 0,  ROTATION_STEP])
    add_binding(("O", "o"), [0, 0, 0, 0, 0, -ROTATION_STEP])

    for key in ("R", "r"):
        visualizer.register_key_callback(ord(key), _reset_callback)
    for key in ("P", "p"):
        visualizer.register_key_callback(ord(key), _print_callback)
    for key in ("C", "c"):
        visualizer.register_key_callback(ord(key), _confirm_callback)

# -------------------------
# Camera helpers for first-person view
# -------------------------
class FirstPersonCameraState:
    def __init__(self):
        self.eye = None
        self.center = None
        self.up = None
        self.R_wc = None   # 3x3
        self.T_wc = None   # 4x4

fp_cam_state = FirstPersonCameraState()

def set_first_person_camera(vis, endo_start_model, endo_end_model, up_hint_model=None, fov_deg=60.0, size_wh=(640, 360)):
    """
    在给定 visualizer 上，将相机设置到第一视角，并记录相机位姿以绘制坐标系。
    """
    def _safe_normalize(v, eps=1e-12):
        n = np.linalg.norm(v)
        if n < eps:
            return np.array([0.0, 0.0, 1.0], dtype=float), False
        return v / n, True

    dir_vec, ok = _safe_normalize(np.array(endo_end_model) - np.array(endo_start_model))
    if not ok:
        dir_vec = np.array([0.0, 0.0, 1.0], dtype=float)

    eye =  endo_end_model + dir_vec*5 # np.array(endo_end_model, dtype=float)
    center = eye + dir_vec

    if up_hint_model is None:
        up = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        up, _ = _safe_normalize(np.array(up_hint_model, dtype=float))

    # 正交化 up
    right = np.cross(dir_vec, up)
    rnorm = np.linalg.norm(right)
    if rnorm < 1e-12:
        tmp = np.array([0.0, 1.0, 0.0], dtype=float)
        right = np.cross(dir_vec, tmp)
    right = right / (np.linalg.norm(right) + 1e-12)
    up = np.cross(right, dir_vec)

    # 相机参数
    ctr = vis.get_view_control()
    params = ctr.convert_to_pinhole_camera_parameters()

    # 内参
    w, h = int(size_wh[0]), int(size_wh[1])
    fx = fy = 0.5 * w / np.tan(np.deg2rad(fov_deg) * 0.5)
    cx = w / 2.0
    cy = h / 2.0
    intr = params.intrinsic
    intr.set_intrinsics(w, h, fx, fy, cx, cy)

    # 外参（世界->相机）
    z_cam = (center - eye); z_cam = z_cam / (np.linalg.norm(z_cam) + 1e-12)
    x_cam = np.cross(z_cam, up); x_cam = x_cam / (np.linalg.norm(x_cam) + 1e-12)
    y_cam = np.cross(z_cam, x_cam)

    R_wc = np.column_stack([x_cam, y_cam, z_cam])
    t_wc = eye
    R_cw = R_wc.T
    t_cw = -R_cw @ t_wc

    extr = np.eye(4)
    extr[:3, :3] = R_cw
    extr[:3, 3] = t_cw
    params.extrinsic = extr

    ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)

    # 记录状态，用于绘制相机坐标系
    fp_cam_state.eye = eye
    fp_cam_state.center = center
    fp_cam_state.up = up
    fp_cam_state.R_wc = R_wc.copy()
    T_wc = np.eye(4)
    T_wc[:3, :3] = R_wc
    T_wc[:3, 3] = t_wc
    fp_cam_state.T_wc = T_wc

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    # 读取模型
    mesh = o3d.io.read_triangle_mesh("HEAD2_hole.stl")
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([1, 0.706, 0])

    mesh_brain = o3d.io.read_triangle_mesh("BRAIN2_hole.stl")
    mesh_brain.compute_vertex_normals()
    mesh_brain.paint_uniform_color([0.7, 0.7, 0.7])
    mesh += mesh_brain

    # STL坐标系显示
    stl_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50, origin=[0, 0, 0])

    # 定义模型坐标系下的 init_frame 目标位姿
    T_model2init = pose_to_transformation_matrix([65, 30, 100, 0, np.pi, -np.pi/2])
    init_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=30, origin=[0, 0, 0])
    init_frame.transform(T_model2init)

    # 加载（如需）EE相对模型：仅用于显示
    ee2model = np.load("./trans_matrix_ee2model.npy")  # 假设为 T_model2ee

    # endobase 相对 ee 的位姿（endobase->ee）
    endobase2ee = pose_to_transformation_matrix([-20, 60, -30, 0, 0, np.pi/2+np.pi])

    # 初始 base->model（可从文件载入）
    base2model_nominal = np.array([
        [-9.82277681e-01, -1.11689625e-01, -1.50519056e-01,  1.96873779e+02],
        [ 1.04528463e-01,  3.40146521e-01, -9.34544886e-01,  1.61664136e+02],
        [ 1.55577501e-01, -9.33716109e-01, -3.22443591e-01,  6.74934005e+02],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00],
    ])

    # 可视化坐标系
    ee = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50, origin=[0, 0, 0])
    ee.transform(ee2model)

    base = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=[0, 0, 0])
    base.transform(transform_base2model @ base2model_nominal)

    endobase = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=[0, 0, 0])
    T_ee2endobase = np.linalg.inv(endobase2ee)
    T_model2endobase = ee2model @ T_ee2endobase
    endobase.transform(T_model2endobase)

    # 线段初值（毫米，假设在 endobase 局部坐标系）
    endo_start = [82.17005947, -169.99474041, 87.86597535]
    endo_end   = [66.77834087,   23.57921231, 104.41069992]
    endo_rcm   = [0.0, 0.0, 0.0]

    # 用于确认时的缓存
    last_endo_end   = endo_end

    line = o3d.geometry.LineSet()
    line.points = o3d.utility.Vector3dVector([endo_start, endo_end])
    line.lines = o3d.utility.Vector2iVector([[0, 1]])
    line.paint_uniform_color([0, 1, 0])

    # -------------------------
    # 创建两个窗口
    # -------------------------
    # 第三视角窗口参数
    win_size_3rd = (960, 540)
    vis3 = o3d.visualization.VisualizerWithKeyCallback()
    vis3.create_window(window_name="Third Person View", width=win_size_3rd[0], height=win_size_3rd[1], left=50, top=50)
    register_pose_keys(vis3)

    opt3 = vis3.get_render_option()
    opt3.background_color = np.asarray([0.5, 0.5, 0.5])
    opt3.point_size = 1.0
    opt3.show_coordinate_frame = True
    opt3.mesh_show_back_face = True

    vis3.add_geometry(mesh)
    # vis3.add_geometry(stl_frame)
    # vis3.add_geometry(init_frame)
    # vis3.add_geometry(ee)
    # vis3.add_geometry(base)
    # vis3.add_geometry(endobase)
    vis3.add_geometry(line)

    ctr3 = vis3.get_view_control()
    ctr3.set_zoom(0.5)
    ctr3.set_front([0, 0, -1])
    ctr3.set_lookat([0, 0, 0])
    ctr3.set_up([0, -1, 0])

    # 第一视角窗口参数
    win_size_1st = (960, 540)
    vis1 = o3d.visualization.VisualizerWithKeyCallback()
    vis1.create_window(window_name="First Person (Endoscope) View", width=win_size_1st[0], height=win_size_1st[1], left=1040, top=50)
    register_pose_keys(vis1)

    opt1 = vis1.get_render_option()
    opt1.background_color = np.asarray([0.1, 0.1, 0.1])
    opt1.point_size = 1.0
    opt1.show_coordinate_frame = False
    opt1.mesh_show_back_face = True

    vis1.add_geometry(mesh)
    vis1.add_geometry(line)

    # Redis
    redis_client = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
    redis_key = "endo_position"
    start_joint_listener()

    old_endobase = None
    old_line = None
    old_line_tip = None
    old_rcm = None

    old_endobase1 = None
    old_line1 = None
    old_line_tip1 = None
    old_rcm1 = None

    # 相机坐标系几何缓存（第一视角与第三视角各自维护）
    cam_axis_geom_vis1 = None
    cam_axis_geom_vis3 = None

    # 初始相机设置一次（避免第一帧未设置）
    base2model = transform_base2model @ base2model_nominal
    # 局部端点 -> 模型
    endo_start_model_init = (T_model2endobase @ np.array([*endo_start, 1.0])).ravel()[:3]
    endo_end_model_init   = (T_model2endobase @ np.array([*endo_end,   1.0])).ravel()[:3]
    set_first_person_camera(vis1, endo_start_model_init, endo_end_model_init, up_hint_model=[0, 0, 1], fov_deg=70.0, size_wh=win_size_1st)

    while True:
        joint = get_latest_joint_angles()

        if joint is None:
            # 即使没有关节角，也驱动两个窗口事件循环
            vis3.poll_events()
            vis3.update_renderer()
            vis1.poll_events()
            vis1.update_renderer()
            time.sleep(0.01)
            continue

        base2model = transform_base2model @ base2model_nominal

        # 读取内镜线段（假设 Redis 以米提供，乘1000转毫米；若本来就是毫米，请去掉 *1000）
        try:
            value = redis_client.get(redis_key)
            if value is not None:
                value = value.decode('utf-8')
                arr = [float(i) * 1000.0 for i in value.split(",")]
                if len(arr) >= 9:
                    endo_start = arr[0:3]
                    endo_end   = arr[3:6]
                    endo_rcm   = arr[6:9]
                    last_endo_end = endo_end
        except Exception as e:
            print("Redis error:", e)

        joint_rad = joint
        T_base2ee = fk(joint_rad)

        T_ee2endobase = np.linalg.inv(endobase2ee)
        T_base2endobase = T_base2ee @ T_ee2endobase
        T_model2endobase = base2model @ T_base2endobase

        # 重建几何（第三视角窗口）
        if old_endobase is not None:
            vis3.remove_geometry(old_endobase, False)
        if old_line is not None:
            vis3.remove_geometry(old_line, False)
        if old_line_tip is not None:
            vis3.remove_geometry(old_line_tip, False)
        if old_rcm is not None:
            vis3.remove_geometry(old_rcm, False)

        endobase_geo = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=[0, 0, 0])
        endobase_geo.transform(T_model2endobase)

        thick_line, tip_line, rcm_point = create_thick_line(endo_start, endo_end, endo_rcm, radius=4.0, color=[1, 0, 0])
        thick_line.transform(T_model2endobase)
        tip_line.transform(T_model2endobase)
        rcm_point.transform(T_model2endobase)

        vis3.add_geometry(rcm_point, False)
        vis3.add_geometry(thick_line, False)
        vis3.add_geometry(tip_line, False)
        # vis3.add_geometry(endobase_geo, False)

        old_endobase = endobase_geo
        old_line = thick_line
        old_line_tip = tip_line
        old_rcm = rcm_point

        # 第一视角窗口也更新几何
        if old_endobase1 is not None:
            vis1.remove_geometry(old_endobase1, False)
        if old_line1 is not None:
            vis1.remove_geometry(old_line1, False)
        if old_line_tip1 is not None:
            vis1.remove_geometry(old_line_tip1, False)
        if old_rcm1 is not None:
            vis1.remove_geometry(old_rcm1, False)

        endobase_geo1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=[0, 0, 0])
        endobase_geo1.transform(T_model2endobase)

        thick_line1, tip_line1, rcm_point1 = create_thick_line(endo_start, endo_end, endo_rcm, radius=4.0, color=[1, 0, 0])
        thick_line1.transform(T_model2endobase)
        tip_line1.transform(T_model2endobase)
        rcm_point1.transform(T_model2endobase)

        vis1.add_geometry(rcm_point1, False)
        vis1.add_geometry(thick_line1, False)
        vis1.add_geometry(tip_line1, False)
        # vis1.add_geometry(endobase_geo1, False)

        old_endobase1 = endobase_geo1
        old_line1 = thick_line1
        old_line_tip1 = tip_line1
        old_rcm1 = rcm_point1

        # 计算 endo_start/end 在模型坐标下的位置，更新第一视角相机
        p_start_model = (T_model2endobase @ np.array([endo_start[0], endo_start[1], endo_start[2], 1.0])).ravel()[:3]
        p_end_model   = (T_model2endobase @ np.array([endo_end[0],   endo_end[1],   endo_end[2],   1.0])).ravel()[:3]

        # up_hint 使用 endobase 的 +Z 轴在模型坐标中的方向
        z_axis_endobase_model = T_model2endobase[:3, :3] @ np.array([0.0, 0.0, 1.0])
        set_first_person_camera(vis1, p_start_model, p_end_model, up_hint_model=z_axis_endobase_model, fov_deg=70.0, size_wh=win_size_1st)

        # 更新/显示相机坐标系（与第一视角相机外参一致）——两个窗口都显示
        if fp_cam_state.T_wc is not None:
            # 第一视角窗口
            if cam_axis_geom_vis1 is not None:
                vis1.remove_geometry(cam_axis_geom_vis1, False)
            cam_axis_geom_vis1 = create_camera_axis_T(fp_cam_state.T_wc, axis_len=cam_axis_len)
            # vis1.add_geometry(cam_axis_geom_vis1, False)

            # 第三视角窗口
            if cam_axis_geom_vis3 is not None:
                vis3.remove_geometry(cam_axis_geom_vis3, False)
            cam_axis_geom_vis3 = create_camera_axis_T(fp_cam_state.T_wc, axis_len=cam_axis_len)
            # vis3.add_geometry(cam_axis_geom_vis3, False)

        # 渲染两个窗口
        vis3.poll_events()
        vis3.update_renderer()
        vis1.poll_events()
        vis1.update_renderer()
        time.sleep(0.01)
       