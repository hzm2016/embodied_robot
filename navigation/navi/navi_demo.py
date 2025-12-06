from realman_command import RMCommand
from realman_kinematics import forward_kinematics as fk
import numpy as np
import cv2
import open3d as o3d
import time
import redis


transform_base2model = np.eye(4)
base2model_nominal = None
TRANSLATION_STEP = 5.0  # millimeters
ROTATION_STEP = np.deg2rad(1.0)  # radians

def pose_to_transformation_matrix(pose):
    """
    Convert pose [x,y,z,rx,ry,rz] to 4x4 transformation matrix
    Args:
        pose: list or numpy array [x,y,z,rx,ry,rz] 
            where rx,ry,rz are rotation angles in radians
            x,y,z are in millimeters
    Returns:
        4x4 homogeneous transformation matrix
    """
    x, y, z = pose[0:3]
    rx, ry, rz = pose[3:6]

    # Calculate rotation matrices for each axis
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])
    
    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    
    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ])

    # Combined rotation matrix
    R = Rz @ Ry @ Rx

    # Create 4x4 transformation matrix
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = [x, y, z]
    
    return T
def transformation_matrix_to_pose(T):
    """
    Convert 4x4 transformation matrix to pose [x,y,z,rx,ry,rz]
    Args:
        T: 4x4 homogeneous transformation matrix
    Returns:
        pose [x,y,z,rx,ry,rz] where rx,ry,rz are rotation angles in radians
    """
    # Extract position
    x = T[0,3]
    y = T[1,3]
    z = T[2,3]
    
    # Extract rotation matrix
    R = T[0:3, 0:3]
    
    # Calculate rotation angles
    ry = np.arctan2(R[0,2], np.sqrt(R[0,0]**2 + R[0,1]**2))
    rx = np.arctan2(-R[1,2]/np.cos(ry), R[2,2]/np.cos(ry))
    rz = np.arctan2(-R[0,1]/np.cos(ry), R[0,0]/np.cos(ry))
    
    return np.array([x, y, z, rx, ry, rz])


def _print_transform_state():
    global base2model_nominal, transform_base2model
    effective = transform_base2model @ base2model_nominal if base2model_nominal is not None else transform_base2model
    pose = transformation_matrix_to_pose(transform_base2model)
    print("transform_base2model delta pose [mm, rad]:", pose)
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

def create_thick_line(start, end, rcm, radius=2.0, color=[1,0,0]):
    """
    Create a thick line using a cylinder
    start: starting point [x,y,z]
    end: ending point [x,y,z]
    radius: thickness of the line
    color: RGB color of the line
    """
    # Create a cylinder
    vector = np.array(end) - np.array(start)
    length = np.linalg.norm(vector)
    direction = vector / length
    
    # Create mesh cylinder
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length)
    
    # Rotate cylinder to point in the right direction
    rotation = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    if not np.allclose(direction, [0, 0, 1]):
        axis = np.cross([0, 0, 1], direction)
        axis_length = np.linalg.norm(axis)
        if axis_length > 0:
            axis = axis / axis_length
            angle = np.arccos(np.dot([0, 0, 1], direction))
            rotation = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
    # create a point
    point = o3d.geometry.TriangleMesh.create_sphere(radius=radius*1.5)
    point.translate(rcm)
    point.paint_uniform_color([0,1,0])
    
    # Transform cylinder to correct position and orientation
    cylinder.rotate(rotation, center=[0, 0, 0])
    cylinder.translate(start + direction * length/2)
    
    # Color the cylinder
    cylinder.paint_uniform_color(color)

    cylinder_tip = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=radius*2)
    cylinder_tip.rotate(rotation, center=[0, 0, 0])
    cylinder_tip.translate(end)
    cylinder_tip.paint_uniform_color([0,0,1])
    
    return cylinder,cylinder_tip,point

def cal_endo2ee(pos):
    '''
    pos-[分别代表内镜伸缩电机0、竖直基座电机(1-2)和水平电机(3-4)，单位mm]
    '''
    ## 定长(mm)，0~4对应坐标系序号
    Ly0 = 60  # Y0方向基座长度
    Lz0 = 70  # Z0方向基座到平台原始距离
    Lx2 = 59  # X2方向34杆间距
    Ly2 = 102  # Y2方向平台到坐标系3距离
    Lz2 = 67  # Z2方向34杆长度
    Lz43 = 53  # Z4反方向到坐标系3存在的机械长度
    L = [Lz43 + pos[0], pos[1] + Lz0, pos[2] + Lz0, Lz2 - pos[3], pos[4]]
    psi12 = np.arctan2(L[1] - L[2], Ly0)
    psi34 = np.arctan2(L[3] - L[4] , Lx2)
    T01 = pose_to_transformation_matrix([0,0,L[1]/1000,0,0,0])
    T12 = pose_to_transformation_matrix([0,0,0,0,np.pi/2,np.pi/2-psi12])
    T23 = pose_to_transformation_matrix([0,Ly2/1000, L[3]/1000,0, 0, 0])
    T34 = pose_to_transformation_matrix([0,0, L[0]/1000,0, np.pi/2 + psi34,0])
    T04 = T01 @ T12 @ T23 @ T34
    return T04

if __name__ == "__main__":

    rm_command = RMCommand()
    rm_command.connect_tcp_socket()
    time.sleep(1)

    mesh = o3d.io.read_triangle_mesh("HEAD2_hole.stl")
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([1, 0.706, 0])
    stl_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50, origin=[0,0,0])

    init_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=30, origin=[0,0,0])
    init_frame.transform(pose_to_transformation_matrix([65,30,100,0,0,0]))
    
    ee2model = np.load("./trans_matrix_ee2model.npy")

    endobase2ee = pose_to_transformation_matrix([-20,60,-30,0,0,np.pi/2]) # 可变
    endobase2model = ee2model @ endobase2ee

    joint = rm_command.read_j()
    joint_rad = [j*np.pi/180 for j in joint]

    base2model_nominal = [[-9.82277681e-01, -1.11689625e-01, -1.50519056e-01,  1.96873779e+02],
    [ 1.04528463e-01,  3.40146521e-01, -9.34544886e-01,  1.61664136e+02],
    [ 1.55577501e-01, -9.33716109e-01, -3.22443591e-01,  6.74934005e+02],
    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00],]

    
    #np.load("./trans_matrix_base2model.npy")
    base2model = transform_base2model @ base2model_nominal
    print("####################")
    print(base2model)

    ## 绘制base、camera、ee坐标系和内镜
    ee = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50, origin=[0,0,0])
    ee.transform(ee2model)
    base = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=[0,0,0])
    base.transform(base2model)
    endobase = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=[0,0,0])
    endobase.transform(endobase2model)

    endo_start= [82.17005947, -169.99474041,   87.86597535]
    endo_end = [66.77834087 , 23.57921231, 104.41069992]
    line = o3d.geometry.LineSet()
    line.points = o3d.utility.Vector3dVector([endo_start,endo_end])
    lines = [[0,1]]
    line.lines = o3d.utility.Vector2iVector(lines)
    line.paint_uniform_color([0,1,0])
    
    path_line = o3d.geometry.LineSet()
    path_line.points = o3d.utility.Vector3dVector([endo_start,endo_end])
    path_lines = [[0,1]]
    path_line.lines = o3d.utility.Vector2iVector(path_lines)
    path_line.paint_uniform_color([0,1,0])

    # Create and setup visualizer
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(width=1280, height=720)

    def register_pose_keys(visualizer):
        def add_binding(keys, delta_pose):
            callback = _make_delta_callback(delta_pose)
            for key in keys:
                visualizer.register_key_callback(ord(key), callback)

        add_binding(("A", "a"), [-TRANSLATION_STEP, 0, 0, 0, 0, 0])
        add_binding(("D", "d"), [TRANSLATION_STEP, 0, 0, 0, 0, 0])
        add_binding(("Q", "q"), [0, TRANSLATION_STEP, 0, 0, 0, 0])
        add_binding(("E", "e"), [0, -TRANSLATION_STEP, 0, 0, 0, 0])
        add_binding(("W", "w"), [0, 0, TRANSLATION_STEP, 0, 0, 0])
        add_binding(("S", "s"), [0, 0, -TRANSLATION_STEP, 0, 0, 0])

        add_binding(("I", "i"), [0, 0, 0, ROTATION_STEP, 0, 0])
        add_binding(("K", "k"), [0, 0, 0, -ROTATION_STEP, 0, 0])
        add_binding(("J", "j"), [0, 0, 0, 0, ROTATION_STEP, 0])
        add_binding(("L", "l"), [0, 0, 0, 0, -ROTATION_STEP, 0])
        add_binding(("U", "u"), [0, 0, 0, 0, 0, ROTATION_STEP])
        add_binding(("O", "o"), [0, 0, 0, 0, 0, -ROTATION_STEP])

        for key in ("R", "r"):
            visualizer.register_key_callback(ord(key), _reset_callback)
        for key in ("P", "p"):
            visualizer.register_key_callback(ord(key), _print_callback)

    register_pose_keys(vis)
    
    # Set rendering options for better visualization
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.5, 0.5, 0.5])  # Gray background
    opt.point_size = 1.0
    opt.show_coordinate_frame = True
    opt.mesh_show_back_face = True 

    vis.add_geometry(mesh)
    vis.add_geometry(stl_frame)
    vis.add_geometry(init_frame)
    vis.add_geometry(ee)
    vis.add_geometry(base)
    vis.add_geometry(endobase)
    vis.add_geometry(line)

    ctr = vis.get_view_control()
    ctr.set_zoom(0.5)
    ctr.set_front([0, 0, -1])
    ctr.set_lookat([0, 0, 0])
    ctr.set_up([0, -1, 0])

    old_endobase= endobase
    old_line = line
    old_line_tip = line
    old_rcm = line
    # Connect to Redis
    redis_client = redis.StrictRedis(host='localhost', port=6379, db=0) # 192.168.1.94
    key = "endo_position"

    while True:
        joint = rm_command.read_j()
        if joint is not None:
            base2model = transform_base2model @ base2model_nominal
            print("####################")
            print(base2model)
            base = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=[0,0,0])
            base.transform(base2model)
            value = redis_client.get(key).decode('utf-8')
            endo_value = [float(i)*1000 for i in value.split(",")]

            endo_start = endo_value[:3]
            endo_end = endo_value[3:6] 
            endo_rcm =  endo_value[6:]
            joint_rad = [j*np.pi/180 for j in joint]
            # time.sleep(1)
            # update end effector in base frame
            ee2base = fk(joint_rad)
            # update endoscopy in base frame
            endobase2base = ee2base@endobase2ee
            # update endoscopy in model frame
            endobase2model = base2model@endobase2base
            # Remove old line
            vis.remove_geometry(old_endobase, False)
            vis.remove_geometry(old_line, False)
            vis.remove_geometry(old_line_tip, False)
            vis.remove_geometry(old_rcm, False)
            # vis.remove_geometry(old_line_tip, False)

            endobase = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=[0,0,0])
            endobase.transform(endobase2model)

            thick_line,tip_line,rcm_point = create_thick_line(endo_start, endo_end, endo_rcm,radius=4.0, color=[1,0,0])
            thick_line.transform(endobase2model)
            tip_line.transform(endobase2model)
            rcm_point.transform(endobase2model)
            # If using first visualization approach:
            # vis.remove_geometry(old_endobase, False)

            vis.add_geometry(rcm_point, False)
            vis.add_geometry(thick_line, False)
            vis.add_geometry(base, False)
            vis.add_geometry(tip_line, False)
            old_endobase = endobase
            old_line = thick_line
            old_line_tip = tip_line
            old_rcm = rcm_point
            vis.poll_events()
            vis.update_renderer()
