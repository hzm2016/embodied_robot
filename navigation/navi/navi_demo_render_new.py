import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import time
import threading
import json
import redis
from realman_kinematics import forward_kinematics as fk

# -------------------------
# Global / Config
# -------------------------
transform_base2model = np.eye(4)  # 增量微调
base2model_nominal = np.eye(4)    # 标定值

TRANSLATION_STEP = 5.0  # mm
ROTATION_STEP = np.deg2rad(1.0)  # rad

T_model2init = None

REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0
JOINT_CHANNEL = "actual_joint_angle"
REDIS_ENDO_KEY = "endo_position"

_joint_lock = threading.Lock()
_latest_joint_deg = None
_joint_thread = None

align_orientation = True  # True: 位姿都对齐；False: 仅位置对齐
cam_axis_len = 40.0

# 显示参数
HEAD_ALPHA = 0.35  # head 不透明度（0完全透明，1完全不透明）
HEAD_COLOR = (1.0, 0.706, 0.0, HEAD_ALPHA)
BRAIN_COLOR = (0.7, 0.7, 0.7, 1.0)
LINE_COLOR = (1.0, 0.0, 0.0, 1.0)
RCM_COLOR = (0.0, 1.0, 0.0, 1.0)
TIP_COLOR = (0.0, 0.0, 1.0, 1.0)
BG_COLOR = (0.5, 0.5, 0.5, 1.0)
BG_COLOR_1ST = (0.1, 0.1, 0.1, 1.0)

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

# -------------------------
# Redis joint subscription
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
# Endoscope helpers
# -------------------------
def create_thick_line_meshes(start, end, rcm, radius=4.0):
    start = np.array(start, dtype=float)
    end = np.array(end, dtype=float)
    rcm = np.array(rcm, dtype=float)
    vec = end - start
    length = float(np.linalg.norm(vec) + 1e-12)
    dirv = vec / (np.linalg.norm(vec) + 1e-12)
    z_axis = np.array([0.0, 0.0, 1.0], dtype=float)

    rot = np.eye(3)
    if not np.allclose(dirv, z_axis):
        axis = np.cross(z_axis, dirv)
        axis_len = np.linalg.norm(axis)
        if axis_len > 1e-12:
            axis = axis / axis_len
            angle = np.arccos(np.clip(np.dot(z_axis, dirv), -1.0, 1.0))
            rot = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)

    cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length)
    cyl.rotate(rot, center=[0, 0, 0])
    cyl.translate(start + dirv * (length / 2.0))

    tip = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=radius * 2.0)
    tip.rotate(rot, center=[0, 0, 0])
    tip.translate(end)

    rcm_ball = o3d.geometry.TriangleMesh.create_sphere(radius=radius * 1.5)
    rcm_ball.translate(rcm)

    return cyl, tip, rcm_ball

# -------------------------
# First-person camera
# -------------------------
class FirstPersonCameraState:
    def __init__(self):
        self.eye = None
        self.center = None
        self.up = None
        self.R_wc = None
        self.T_wc = None

fp_cam_state = FirstPersonCameraState()

def set_first_person_camera_in_scene(scene: rendering.Open3DScene, endo_start_model, endo_end_model, up_hint_model=None, fov_deg=70.0):
    def _safe_normalize(v, eps=1e-12):
        n = np.linalg.norm(v)
        if n < eps:
            return np.array([0.0, 0.0, 1.0], dtype=float), False
        return v / n, True

    endo_start_model = np.array(endo_start_model, dtype=float)
    endo_end_model = np.array(endo_end_model, dtype=float)

    dir_vec, ok = _safe_normalize(endo_end_model - endo_start_model)
    if not ok:
        dir_vec = np.array([0.0, 0.0, 1.0], dtype=float)

    eye = endo_end_model + dir_vec * 5.0
    center = eye + dir_vec

    if up_hint_model is None:
        up = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        up, _ = _safe_normalize(np.array(up_hint_model, dtype=float))

    right = np.cross(dir_vec, up)
    if np.linalg.norm(right) < 1e-12:
        right = np.cross(dir_vec, np.array([0.0, 1.0, 0.0]))
    right = right / (np.linalg.norm(right) + 1e-12)
    up = np.cross(right, dir_vec)

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

    cam = scene.camera
    cam.set_projection(fov_deg, 1.0, 0.1, 5000.0, rendering.Camera.FovType.Vertical)
    cam.set_view_matrix(extr)

    fp_cam_state.eye = eye
    fp_cam_state.center = center
    fp_cam_state.up = up
    fp_cam_state.R_wc = R_wc.copy()
    T_wc = np.eye(4)
    T_wc[:3, :3] = R_wc
    T_wc[:3, 3] = t_wc
    fp_cam_state.T_wc = T_wc

# -------------------------
# App UI
# -------------------------
class DualViewApp:
    def __init__(self):
        # Redis
        self.redis_client = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
        # Kinematics constant
        self.endobase2ee = pose_to_transformation_matrix([-20, 60, -30, 0, 0, np.pi/2 + np.pi])

        # Load meshes
        self.mesh_head = o3d.io.read_triangle_mesh("HEAD2_hole.stl")
        self.mesh_head.compute_vertex_normals()

        self.mesh_brain = o3d.io.read_triangle_mesh("BRAIN2_hole.stl")
        self.mesh_brain.compute_vertex_normals()

        # Init frame in model
        global T_model2init
        T_model2init = pose_to_transformation_matrix([65, 30, 100, 0, np.pi, -np.pi / 2])
        self.init_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=30, origin=[0, 0, 0])
        self.init_frame.transform(T_model2init)

        # EE to model (display only)
        self.ee2model = np.load("./trans_matrix_ee2model.npy")
        self.ee_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50, origin=[0, 0, 0])
        self.ee_frame.transform(self.ee2model)

        # Base & endobase frames
        self.base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=[0, 0, 0])

        T_ee2endobase = np.linalg.inv(self.endobase2ee)
        T_model2endobase = self.ee2model @ T_ee2endobase
        self.endobase_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=[0, 0, 0])
        self.endobase_frame.transform(T_model2endobase)

        # Endoscope initial (in endobase local, mm)
        self.endo_start = [82.17005947, -169.99474041, 87.86597535]
        self.endo_end   = [66.77834087,   23.57921231, 104.41069992]
        self.endo_rcm   = [0.0, 0.0, 0.0]
        self.last_endo_end = self.endo_end

        # GUI init
        self.app = gui.Application.instance
        self.app.initialize()

        self.window = gui.Application.instance.create_window("Digital Twin - Dual View (O3D GUI)", 1920, 1080)
        em = self.window.theme.font_size

        self.panel = gui.Horiz(0, gui.Margins(4*em, 1*em, 4*em, 1*em))
        self.window.add_child(self.panel)

        # Left: third-person scene
        self.scene3 = rendering.Open3DScene(self.window.renderer)
        self.scene3_widget = gui.SceneWidget()
        self.scene3_widget.scene = self.scene3
        self.scene3_widget.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)
        self.scene3.set_background(BG_COLOR)
        self.panel.add_child(self.scene3_widget)

        # Right: first-person scene
        self.scene1 = rendering.Open3DScene(self.window.renderer)
        self.scene1_widget = gui.SceneWidget()
        self.scene1_widget.scene = self.scene1
        self.scene1_widget.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)
        self.scene1.set_background(BG_COLOR_1ST)
        self.panel.add_child(self.scene1_widget)

        # Materials: 仅使用兼容字段
        self.mat_head = rendering.MaterialRecord()
        self.mat_head.shader = "defaultLit"
        self.mat_head.base_color = HEAD_COLOR
        if hasattr(rendering.MaterialRecord, "AlphaMode"):
            self.mat_head.alpha_mode = rendering.MaterialRecord.AlphaMode.Blend
        else:
            self.mat_head.alpha_mode = "BLEND"
        if hasattr(self.mat_head, "double_sided"):
            self.mat_head.double_sided = True  # 透明常见需求

        self.mat_brain = rendering.MaterialRecord()
        self.mat_brain.shader = "defaultLit"
        self.mat_brain.base_color = BRAIN_COLOR
        if hasattr(rendering.MaterialRecord, "AlphaMode"):
            self.mat_brain.alpha_mode = rendering.MaterialRecord.AlphaMode.Opaque
        else:
            self.mat_brain.alpha_mode = "OPAQUE"

        self.mat_line = rendering.MaterialRecord()
        self.mat_line.shader = "defaultLit"
        self.mat_line.base_color = LINE_COLOR
        if hasattr(rendering.MaterialRecord, "AlphaMode"):
            self.mat_line.alpha_mode = rendering.MaterialRecord.AlphaMode.Opaque
        else:
            self.mat_line.alpha_mode = "OPAQUE"

        self.mat_tip = rendering.MaterialRecord()
        self.mat_tip.shader = "defaultLit"
        self.mat_tip.base_color = TIP_COLOR

        self.mat_rcm = rendering.MaterialRecord()
        self.mat_rcm.shader = "defaultLit"
        self.mat_rcm.base_color = RCM_COLOR

        # Add static geometries
        self.add_static_geometry()

        # Key shortcuts
        self.window.set_on_key(self._on_key)

        # Timer for updates
        self.timer = gui.Timer(0.01, self._on_timer)
        self.window.set_on_close(self._on_close)
        self.timer.start()

        # Redis
        start_joint_listener()

    def add_static_geometry(self):
        # Head (semi-transparent)
        self.scene3.add_geometry("head", self.mesh_head, self.mat_head)
        self.scene1.add_geometry("head", self.mesh_head, self.mat_head)

        # Brain (opaque)
        self.scene3.add_geometry("brain", self.mesh_brain, self.mat_brain)
        self.scene1.add_geometry("brain", self.mesh_brain, self.mat_brain)

        # Setup cameras
        bounds = self.scene3.bounding_box
        self.scene3.setup_camera(60.0, bounds, bounds.get_center())
        # 模拟原先第三视角参数
        self.scene3.camera.look_at([0, 0, 0], [0, 0, -800], [0, -1, 0])
        self.scene1.setup_camera(70.0, bounds, bounds.get_center())

        # Dynamic endoscope shapes
        self.update_endoscope_meshes(np.eye(4))

        # First-person initial set
        T_ee2endobase = np.linalg.inv(self.endobase2ee)
        T_model2endobase = self.ee2model @ T_ee2endobase
        p_start_model = (T_model2endobase @ np.array([*self.endo_start, 1.0])).ravel()[:3]
        p_end_model   = (T_model2endobase @ np.array([*self.endo_end,   1.0])).ravel()[:3]
        set_first_person_camera_in_scene(self.scene1, p_start_model, p_end_model, up_hint_model=[0, 0, 1], fov_deg=70.0)

    def update_endoscope_meshes(self, T_model2endobase):
        # Remove previous if exists
        for scene in (self.scene3, self.scene1):
            for name in ("endo_cyl", "endo_tip", "endo_rcm"):
                if scene.has_geometry(name):
                    scene.remove_geometry(name)

        cyl, tip, rcm_ball = create_thick_line_meshes(self.endo_start, self.endo_end, self.endo_rcm, radius=4.0)
        cyl.transform(T_model2endobase)
        tip.transform(T_model2endobase)
        rcm_ball.transform(T_model2endobase)

        self.scene3.add_geometry("endo_cyl", cyl, self.mat_line)
        self.scene3.add_geometry("endo_tip", tip, self.mat_tip)
        self.scene3.add_geometry("endo_rcm", rcm_ball, self.mat_rcm)

        self.scene1.add_geometry("endo_cyl", cyl, self.mat_line)
        self.scene1.add_geometry("endo_tip", tip, self.mat_tip)
        self.scene1.add_geometry("endo_rcm", rcm_ball, self.mat_rcm)

    def _on_key(self, event):
        if event.type == gui.KeyEvent.DOWN:
            k = event.key
            if k in (ord('A'), ord('a')):
                _apply_keyboard_transform([-TRANSLATION_STEP, 0, 0, 0, 0, 0])
            elif k in (ord('D'), ord('d')):
                _apply_keyboard_transform([ TRANSLATION_STEP, 0, 0, 0, 0, 0])
            elif k in (ord('Q'), ord('q')):
                _apply_keyboard_transform([0,  TRANSLATION_STEP, 0, 0, 0, 0])
            elif k in (ord('E'), ord('e')):
                _apply_keyboard_transform([0, -TRANSLATION_STEP, 0, 0, 0, 0])
            elif k in (ord('W'), ord('w')):
                _apply_keyboard_transform([0, 0,  TRANSLATION_STEP, 0, 0, 0])
            elif k in (ord('S'), ord('s')):
                _apply_keyboard_transform([0, 0, -TRANSLATION_STEP, 0, 0, 0])
            elif k in (ord('I'), ord('i')):
                _apply_keyboard_transform([0, 0, 0,  ROTATION_STEP, 0, 0])
            elif k in (ord('K'), ord('k')):
                _apply_keyboard_transform([0, 0, 0, -ROTATION_STEP, 0, 0])
            elif k in (ord('J'), ord('j')):
                _apply_keyboard_transform([0, 0, 0, 0,  ROTATION_STEP, 0])
            elif k in (ord('L'), ord('l')):
                _apply_keyboard_transform([0, 0, 0, 0, -ROTATION_STEP, 0])
            elif k in (ord('U'), ord('u')):
                _apply_keyboard_transform([0, 0, 0, 0, 0,  ROTATION_STEP])
            elif k in (ord('O'), ord('o')):
                _apply_keyboard_transform([0, 0, 0, 0, 0, -ROTATION_STEP])
            elif k in (ord('R'), ord('r')):
                _reset_transform()
            elif k in (ord('P'), ord('p')):
                _print_transform_state()
            elif k in (ord('C'), ord('c')):
                self._confirm_alignment()
            else:
                return gui.Widget.EventCallbackResult.IGNORED
            return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED

    def _confirm_alignment(self):
        global base2model_nominal, transform_base2model, T_model2init
        joint = get_latest_joint_angles()
        if joint is None:
            print("Confirm failed: no joint data from Redis subscriber.")
            return
        joint_rad = joint
        T_base2ee = fk(joint_rad)

        T_ee2endobase = np.linalg.inv(self.endobase2ee)
        T_base2endobase = T_base2ee @ T_ee2endobase

        p_end = np.array(self.last_endo_end, dtype=float)
        T_endobase2endoend = np.eye(4)
        T_endobase2endoend[:3, :3] = np.eye(3)
        T_endobase2endoend[:3, 3] = p_end

        T_base2endoend = T_base2endobase @ T_endobase2endoend

        if align_orientation:
            T_model2endoend_tgt = T_model2init.copy()
        else:
            base2model_eff = transform_base2model @ base2model_nominal
            T_model2endoend_cur = base2model_eff @ T_base2endoend
            T_model2endoend_tgt = np.eye(4)
            T_model2endoend_tgt[:3, :3] = T_model2endoend_cur[:3, :3]
            T_model2endoend_tgt[:3, 3] = T_model2init[:3, 3]

        T_base2model_new = T_model2endoend_tgt @ np.linalg.inv(T_base2endoend)
        base2model_nominal = T_base2model_new
        transform_base2model = np.eye(4)
        print("[Confirm] Aligned endo_end to init_frame with endobase-aligned orientation.")
        _print_transform_state()

    def _on_timer(self):
        # Read Redis endoscope line
        try:
            value = self.redis_client.get(REDIS_ENDO_KEY)
            if value is not None:
                value = value.decode('utf-8')
                arr = [float(i) * 1000.0 for i in value.split(",")]  # 如果单位已是毫米，请去掉 *1000
                if len(arr) >= 9:
                    self.endo_start = arr[0:3]
                    self.endo_end   = arr[3:6]
                    self.endo_rcm   = arr[6:9]
                    self.last_endo_end = self.endo_end
        except Exception as e:
            print("Redis read error:", e)

        joint = get_latest_joint_angles()
        if joint is None:
            return

        joint_rad = joint
        T_base2ee = fk(joint_rad)

        T_ee2endobase = np.linalg.inv(self.endobase2ee)
        T_base2endobase = T_base2ee @ T_ee2endobase

        base2model = transform_base2model @ base2model_nominal
        T_model2endobase = base2model @ T_base2endobase

        # Update endoscope meshes
        self.update_endoscope_meshes(T_model2endobase)

        # Update first-person camera
        p_start_model = (T_model2endobase @ np.array([self.endo_start[0], self.endo_start[1], self.endo_start[2], 1.0])).ravel()[:3]
        p_end_model   = (T_model2endobase @ np.array([self.endo_end[0],   self.endo_end[1],   self.endo_end[2],   1.0])).ravel()[:3]
        z_axis_endobase_model = T_model2endobase[:3, :3] @ np.array([0.0, 0.0, 1.0])
        set_first_person_camera_in_scene(self.scene1, p_start_model, p_end_model, up_hint_model=z_axis_endobase_model, fov_deg=70.0)

    def _on_close(self):
        if self.timer is not None:
            self.timer.stop()
        return True

def main():
    global base2model_nominal
    base2model_nominal = np.array([
        [-9.82277681e-01, -1.11689625e-01, -1.50519056e-01,  1.96873779e+02],
        [ 1.04528463e-01,  3.40146521e-01, -9.34544886e-01,  1.61664136e+02],
        [ 1.55577501e-01, -9.33716109e-01, -3.22443591e-01,  6.74934005e+02],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00],
    ])
    app = DualViewApp()
    gui.Application.instance.run()

if __name__ == "__main__":
    main()