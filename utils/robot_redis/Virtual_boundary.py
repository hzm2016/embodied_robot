import numpy as np
import open3d as o3d
from math import sin, cos, tan, atan, pi



def checkInBoundary(point, orient, boundary):
    orient = orient.T
    # boundary.estimate_normals()
    # mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(boundary, depth=8)
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(boundary)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)

    ray = np.hstack((np.matrix(point), orient))
    ray = o3d.core.Tensor(ray, dtype=o3d.core.Dtype.Float32)
    t_intersection = scene.count_intersections(ray).numpy()
    print(t_intersection)
    ans = scene.list_intersections(ray)
    t_hit = scene.cast_rays(ray)['t_hit'].numpy()
    # print(ans['primitive_uvs'])
    # return t_intersection%2 == 1
    return t_intersection != 0

if __name__ == '__main__':
    filepath="./HALFHEAD_hole_real.stl"
    mesh = o3d.io.read_triangle_mesh(filepath)
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=2).translate([2, 2, 2])
    
    # pcd = sphere.sample_points_poisson_disk(4000)
    
    orient = np.matrix([[1], [1], [1]])
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[3, 3, 3])
    o3d.visualization.draw_geometries([sphere, frame],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])
    if checkInBoundary([3, 3, 3], orient, sphere):
        print("not out of bound")
    else:
        print("out of bound")
