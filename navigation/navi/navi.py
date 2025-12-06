from realman_command import RMCommand
from realman_kinematics import forward_kinematics as fk
import numpy as np
import cv2
import open3d as o3d
import time

def pose_to_transformation_matrix(pose):
    """
    Convert pose [x,y,z,rx,ry,rz] to 4x4 transformation matrix
    Args:
        pose: list or numpy array [x,y,z,rx,ry,rz] 
            where rx,ry,rz are rotation angles in radians
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

def create_thick_line(start, end, radius=2.0, color=[1,0,0]):
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
    
    # Transform cylinder to correct position and orientation
    cylinder.rotate(rotation, center=[0, 0, 0])
    cylinder.translate(start + direction * length/2)
    
    # Color the cylinder
    cylinder.paint_uniform_color(color)
    
    return cylinder

if __name__ == "__main__":
    # load 3d model
    model_path = "/home/mingcong/projects/p9/regesiteration/data/"
    mesh = o3d.io.read_triangle_mesh(model_path + "headtest/stl/head.stl")

    # model to camera
    model2camera = np.load("/home/mingcong/projects/p9/regesiteration/data/trans_matrix_head2camera.npy")
    model2camera[:3,3] = model2camera[:3,3]
    # print(model2camera)
    camera2model = np.linalg.inv(model2camera)
    print(camera2model)

    pcd = mesh.sample_points_uniformly(number_of_points=30000)
    downpcd = pcd.voxel_down_sample(voxel_size=2/1000)
    # draw initial position
    # camera in end effector frame
    camera2ee = pose_to_transformation_matrix([0.08,-0.04,0.04,-np.pi/2,0,0])
    print("camera2ee")
    print(camera2ee)
    # end effector in model frame
    ee2camera = np.linalg.inv(camera2ee)
    ee2model = camera2model @ ee2camera

    # endoscopy to end effector
    endo2ee = pose_to_transformation_matrix([0,220/1000,200/1000,-np.pi/2,0,0])
    # endoscopy to model
    endo2model = ee2model @ endo2ee
    print("endo2model")
    print(endo2model)
    # end effector in base frame
    joint = [88.438, -34.931, -27.111, 2.158, 10.254, 98.338, -122.634]
    joint_rad = [j*np.pi/180 for j in joint]
    ee2base = fk(joint_rad)
    print("ee2base")
    print(ee2base)
    # camera in base frame
    camera2base = ee2base @ camera2ee
    print("camera2base")
    print(camera2base)
    # model in base frame
    model2base = camera2base @ model2camera
    print("model2base")
    print(model2base)
    # base in model frame
    base2model = np.linalg.inv(model2base)

    # draw model origin
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20, origin=[0,0,0])
    # draw camera
    camera = o3d.geometry.TriangleMesh.create_coordinate_frame(size=30, origin=[0,0,0])
    camera2model_draw = np.copy(camera2model)
    camera2model_draw[:3,3] = camera2model_draw[:3,3]
    camera.transform(camera2model_draw)
    # draw end effector
    ee = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20, origin=[0,0,0])
    ee2model_draw = np.copy(ee2model)
    ee2model_draw[:3,3] = ee2model_draw[:3,3]
    ee.transform(ee2model_draw)
    # draw base
    base = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20, origin=[0,0,0])
    base2model_draw = np.copy(base2model)
    base2model_draw[:3,3] = base2model_draw[:3,3]
    base.transform(base2model_draw)
    # draw the endoscopy as a line in z axis
    line = o3d.geometry.LineSet()
    points = np.array([[0,0,0],[0,0,-240]])
    lines = [[0,1]]
    line.points = o3d.utility.Vector3dVector(points)
    line.lines = o3d.utility.Vector2iVector(lines)
    line.paint_uniform_color([1,0,0])
    endo2model_draw = np.copy(endo2model)
    endo2model_draw[:3,3] = endo2model_draw[:3,3]
    line.transform(endo2model_draw)
    # draw the model
    # o3d.visualization.draw_geometries([downpcd,camera,ee,origin,base,line])

    # Create and setup visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1280, height=720)
    
    # Set rendering options for better visualization
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.5, 0.5, 0.5])  # Gray background
    opt.point_size = 1.0
    opt.show_coordinate_frame = True
    print("1")
    # Add initial geometries
    vis.add_geometry(downpcd)
    vis.add_geometry(camera)
    vis.add_geometry(ee)
    vis.add_geometry(origin)
    vis.add_geometry(base)
    vis.add_geometry(line)
    print("2")
    # Set initial camera viewpoint
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    ctr.set_front([0, 0, -1])
    ctr.set_lookat([0, 0, 0])
    ctr.set_up([0, -1, 0])
    print("3")


    rm_command = RMCommand()
    rm_command.connect_tcp_socket()
    # move robot to the initial position
    rm_command.move_j([88.438, -34.931, -27.111, 2.158, 10.254, 98.338, -122.634],5)
    time.sleep(1)
    
     # Initialize old_line for removal
    old_line = line

    while True:
        joint = rm_command.read_j()
        if joint is not None:
            joint_rad = [j*np.pi/180 for j in joint]
            print(joint_rad)
            # time.sleep(1)
            # update end effector in base frame
            ee2base = fk(joint_rad)
            # update endoscopy in base frame
            endo2base = ee2base@endo2ee
            # update endoscopy in model frame
            endo2model = base2model@endo2base
            print("endo2model3")
            print(endo2model)
            # Remove old line
            vis.remove_geometry(old_line, False)
            
           # Create thick line
            start_point = np.array([0, 0, 0])
            end_point = np.array([0, 0, -240])
            thick_line = create_thick_line(start_point, end_point, radius=4.0, color=[0,1,0])
            
            # Transform the line
            endo2model_draw = np.copy(endo2model)
            endo2model_draw[:3,3] = endo2model_draw[:3,3]
            thick_line.transform(endo2model_draw)
            
            # If using first visualization approach:
            vis.remove_geometry(old_line, False)
            vis.add_geometry(thick_line, False)
            old_line = thick_line
            
            vis.poll_events()
            vis.update_renderer()

            # o3d.visualization.draw_geometries([downpcd,camera,ee,origin,base,line])
        

    # # end effector to model
    # ee2model = np.linalg.inv(model2camera) @ ee2camera
    # # endoscopy to end effector
    # endo2ee = pose_to_transformation_matrix([0,240,210,-np.pi/2,0,0])
    # # endoscopy to model
    # endo2model = ee2model @ endo2ee
    # # end effector in base frame
    # joint = [88.438, -34.931, -27.111, 2.158, 10.254, 98.338, -122.634]
    # joint_rad = [j*np.pi/180 for j in joint]
    # ee2base = fk(joint_rad)
    # print("ee2base")
    # print(ee2base)
    # print([j*180/np.pi for j in transformation_matrix_to_pose(ee2base)[3:6]])
    # # print(ee2camera)
    # # camera in end effector frame
    # camera2ee = np.linalg.inv(ee2camera)
    # # camera in base frame
    # camera2base = ee2base @ camera2ee
    # print("camera2base")
    # print(camera2base)
    # print([j*180/np.pi for j in transformation_matrix_to_pose(camera2base)[3:6]])
    # # model in base frame
    # model2base = camera2model @ camera2base
    # print("model2base")
    # print(model2base)

    # ee2base2 = camera2ee @ ee2base
    # print("ee2base2")
    # print(ee2base2)

    # print(transformation_matrix_to_pose(model2base)[:3],[j*180/np.pi for j in transformation_matrix_to_pose(model2base)[3:6]])
    # # base to model
    # base2model = np.linalg.inv(model2base)
    # print("base2model")
    # print(base2model)

   
    # # draw camera
    # camera = o3d.geometry.TriangleMesh.create_coordinate_frame(size=30, origin=[0,0,0])
    # camera.transform(camera2model)
    # # draw end effector
    # ee = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20, origin=[0,0,0])
    # ee.transform(ee2model)
    # # draw endoscopy
    # endo = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0,0,0])
    # endo.transform(endo2model)
    # # draw the robot base
    # base = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20, origin=[0,0,0])
    # base.transform(base2model)

    # # draw the origin coordinate
    # origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20, origin=[0,0,0])
    
    # # draw the endoscopy as a line in z axis
    # line = o3d.geometry.LineSet()
    # points = np.array([[0,0,0],[0,0,-240]])
    # lines = [[0,1]]
    # line.points = o3d.utility.Vector3dVector(points)
    # line.lines = o3d.utility.Vector2iVector(lines)
    # line.paint_uniform_color([1,0,0])
    # line.transform(endo2model)

    # # display the model
    # o3d.visualization.draw_geometries([downpcd,camera,ee,endo,line,origin,base])

