from realman_command import RMCommand
from realman_kinematics import forward_kinematics as fk
import numpy as np
import cv2
import open3d as o3d
import time
import redis

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
    # move robot to the initial position
    # rm_command.move_j([88.438, -34.931, -27.111, 2.158, 10.254, 98.338, -122.634],5)
    time.sleep(1)

    ## load 3d model
    # model_path = "/home/mingcong/projects/p9/regesiteration/data/"
    mesh = o3d.io.read_triangle_mesh("demo_HEAD.stl")
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([1, 0.706, 0])

    model_path = "/home/mingcong/projects/p9/regesiteration/data/"
    mesh2 = o3d.io.read_triangle_mesh(model_path + "headtest/stl/head.stl")
   
    # o3d.visualization.draw_geometries([mesh])
    # pcd = mesh.sample_points_uniformly(number_of_points=800000)
    # downpcd = pcd.voxel_down_sample(voxel_size=2/1000)
    ## T matrix between camera, model, end effector, base
    # model2camera需要重新标定，或者model和camera都以base为基准标定
    # Question:是否应该以base为基准标定，因为base是固定的，而camera随着endoscopy移动，model病人可能会移动


    model2camera = np.load("trans_matrix.npy")
    print(model2camera)
    # print(np.load("/home/mingcong/projects/p9/regesiteration/data/trans_matrix_head2camera.npy"))
    # exit(0)
    model2camera[:3,3] = model2camera[:3,3]*100
    camera2model = np.linalg.inv(model2camera) # 可变?
    ee2camera = pose_to_transformation_matrix([50,-60,-30,np.pi/2,0,np.pi/2]) # 不可变，以远离手术台和相机的底角为坐标轴
    camera2ee = np.linalg.inv(ee2camera)
    ee2model = camera2model @ ee2camera

    joint = rm_command.read_j()
    print(joint)
    # joint_rad = [-2.120208522030192, -0.7157071663653146, 2.308128122592421, -0.3613006084553462, -0.6607816548050531, 1.6056156453721835, -0.6343573699298589]
    joint_rad = [j*np.pi/180 for j in joint]
    ee2base = fk(joint_rad) # 可变
    camera2base = ee2base @ camera2ee
    model2base = camera2base @ model2camera
    base2model = np.linalg.inv(model2base)

    endobase2ee = pose_to_transformation_matrix([0,0,30,0,0,np.pi/2]) # 可变
    endobase2model = ee2model @ endobase2ee
    # end effector in base frame
    # joint = [88.438, -34.931, -27.111, 2.158, 10.254, 98.338, -122.634]
    joint = rm_command.read_j()
    joint_rad = [j*np.pi/180 for j in joint]
    print("####################3")
    print(joint_rad)
    

    ## 绘制base、camera、ee坐标系和内镜
    # origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20, origin=[0,0,0])
    camera = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20, origin=[0,0,0])
    camera.transform(camera2model)
    ee = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50, origin=[0,0,0])
    ee.transform(ee2model)
    base = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=[0,0,0])
    base.transform(base2model)
    endobase = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=[0,0,0])
    endobase.transform(endobase2model)
    # endo start and end
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
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1280, height=720)
    
    # Set rendering options for better visualization
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.5, 0.5, 0.5])  # Gray background
    opt.point_size = 1.0
    opt.show_coordinate_frame = True
    opt.mesh_show_back_face = True 
    print("1")
    # Add initial geometries
    vis.add_geometry(camera)
    vis.add_geometry(mesh2)
    vis.add_geometry(mesh)
    # vis.add_geometry(path_line)
    vis.add_geometry(ee)
    # vis.add_geometry(base)
    # vis.add_geometry(endobase)
    vis.add_geometry(line)
    print("2")
    # Set initial camera viewpoint
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    ctr.set_front([0, 0, -1])
    ctr.set_lookat([0, 0, 0])
    ctr.set_up([0, -1, 0])
    print("3")


    
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

            value = redis_client.get(key).decode('utf-8')
            endo_value = [float(i)*1000 for i in value.split(",")]
            print(endo_value)

            endo_start = endo_value[:3]
            endo_end = endo_value[3:6] 
            endo_rcm =  endo_value[6:]
            print(endo_start, endo_end)
            joint_rad = [j*np.pi/180 for j in joint]
            print(joint_rad)
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
 
            print("-----------")
            print(endo_start, endo_end)
            thick_line,tip_line,rcm_point = create_thick_line(endo_start, endo_end, endo_rcm,radius=4.0, color=[1,0,0])
            thick_line.transform(endobase2model)
            tip_line.transform(endobase2model)
            rcm_point.transform(endobase2model)
            # If using first visualization approach:
            vis.remove_geometry(old_endobase, False)
            vis.remove_geometry(old_line, False)
            vis.add_geometry(rcm_point, False)
            vis.add_geometry(thick_line, False)
            # vis.add_geometry(endobase, False)
            vis.add_geometry(tip_line, False)
            old_endobase = endobase
            old_line = thick_line
            old_line_tip = tip_line
            old_rcm = rcm_point
            vis.poll_events()
            vis.update_renderer()
