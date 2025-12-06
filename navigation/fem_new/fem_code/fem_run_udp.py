import os
import sys
import time
import pickle
import numpy as np
import scipy.sparse as sp
import scipy.linalg as linalg
from qpsolvers import solve_qp
import trimesh
import traceback
# import skeletor
from numba import jit
# from PIL import Image
# from objloader import objloader
# import pycuda.cumath as cumath
# import pycuda.autoinit
import pycuda.driver as cuda
from fem_code.fem_Exclude_Function import fa_vector
from fem_code.fem_cuda_SourceModule import get_cuda_SourceModule
from fem_code.fem_cuda_MatrixStruct import MatrixStruct
from fem_code.fem_function import getnode, TransformMatrix, StiffnessMatrix
from fem_code.RotationEularQuaternion import rotationMatrixToEulerAngles, eulerAnglesToRotationMatrix, \
    rotateToQuaternion, QuaternionToAxialAngle

os.chdir(os.path.split(os.path.realpath(sys.argv[0]))[0])


# @jit(nopython=True)
def set_gravity(node_number, rou, A, g, L, T, f):
    # g = 9.8 # np.array([0, 0, 9.8])
    for i in range(node_number - 1):
        force = 0.5 * rou * g * A * L
        f[6 * i + 2] = force
        f[6 * i + 8] = force
        # 这里默认是选取了z轴正方向为重力方向，如果不是，需要修改T的选取
        torgue = np.dot(T[i, 0:3, 0:3], np.transpose(
            np.array([0, rou / 12 * g * A * T[i, 2, 2] * L ** 2, rou / 12 * g * A * T[i, 2, 1] * L ** 2],
                     dtype=np.float32)))
        f[6 * i + 3:6 * i + 6] = torgue
        f[6 * i + 9:6 * i + 12] = -torgue
    return f


class FEM(object):
    def __init__(self, shared_mem, velocity_InsertionUnit, shared_camera_matrix, mod, trans_matrix, tip_matrix,
                 shared_draw, mode, shared_location, Is_warm_start, shared_udp):
        self.dirs_index = 0
        self.data_time_stream = ''  
        self.Is_save_data = True
        # trans_matrix 模型坐标转深度坐标
        # tip_matrix 末端
        self.hGimbalAngle = np.zeros(2, dtype=np.float32)
        self.hGimbalAngle_copy = np.zeros(2, dtype=np.float32)
        tip_matrix = np.array(tip_matrix, dtype=np.float32)
        trans_matrix = np.array(trans_matrix, dtype=np.float32)
        shared_camera_matrix['trans_matrix'] = trans_matrix
        # trans_matrix = np.transpose(trans_matrix)
        self.shared_camera_matrix = shared_camera_matrix
        self.velocity_InsertionUnit = velocity_InsertionUnit

        # read path
        with open('/home/p9/Projects/embodiedrobot/navigation/data/makehole.txt') as f:
            folderpath = f.readline().replace('\r', '').replace('\n', '')
            print("-------------------------------------")
            print(folderpath)

        objPath = folderpath + '/stl/BRAIN2_hole.stl'
        # objPath = './model/brain+hole.stl'
        obj = trimesh.load(objPath)

        self.x = np.array(obj.vertices, dtype=np.float32)  # 读取点
        self.xtriangle = np.array(obj.faces, dtype=np.int32)  # 读取面
        # 旋转90度
        # cx = linalg.expm(np.cross(np.eye(3), [-1., 0., 0.] / linalg.norm([1, 0, 0]) * 3.1415926535 / 2))
        # self.x = self.x @ np.transpose(cx)
        # 单位改成mm
        # self.x *= 10
        try:
            self.x = self.x @ np.transpose(trans_matrix[0:3, 0:3])
            self.x += 1000 * np.transpose(trans_matrix[0:3, 3])
        except:
            print('trans_matrix is None', trans_matrix)
        try:
            self.topnode = np.zeros((2, 3), dtype=np.float32)
            self.topnode[1] = np.transpose(1000 * tip_matrix[0:3, 3])
            self.topnode[0] = self.topnode[1] - 300 * np.transpose(tip_matrix[0:3, 0])

        except:
            print('tip_matrix is None', tip_matrix)
            # 注意topnode的上下两端不能有参数一样，不能平行于xyz轴
            self.topnode = np.array([[-40., 10., 310.], [-35., 15., 67.]])
            # 10倍是厘米变成mm
            # self.topnode = 10 * np.array([[-0.916783988 + 40 * 0.0104587, 9.344343185 + 40 * 0.98568391, -1.924159646 + 40 * (-0.16827909)], [-0.916783988, 9.344343185, -1.924159646]])

        node_num = 100
        self.gNode, self.gElement = getnode(self.topnode, node_num, True)
        self.gBC1 = np.array(
            [[node_num - 80, 1, 0.0], [node_num - 80, 2, 0.0], [node_num - 80, 3, 0.0], [node_num - 80, 4, 0.0],
             [node_num - 80, 5, 0.0], [node_num - 80, 6, 0.0], [node_num - 20, 1, 0.0], [node_num - 20, 2, 0.0],
             [node_num - 20, 3, 0.0], [node_num - 20, 4, 0.0], [node_num - 20, 5, 0.0], [node_num - 20, 6, 0.0]])
        # 参数还需要调整
        # 管子材料，几个管子用几个，顺序从外到里,从后往前
        self.gMaterial = np.array([[51000.0, 19.233, 9.3, 1.010e-6],  # 坚硬外鞘,管子号：无，为基本不可变形的大硬度刚体管
                                   [510.0, 19.233, 9.3, 1.010e-6],  # 可弯曲鞘，管子号：２，杨氏模量N/mm^2，惯性矩mm^4，横截面积mm^2，密度
                                   [307.0, 19.233, 9.3, 1.010e-6],  # 插入部管子，管子号：1， 杨氏模量N/mm^2，惯性矩mm^4，横截面积mm^2，密度
                                   [107.0, 19.233, 9.3, 1.010e-6]])  # 可弯曲插入部

        self.ResultNode = np.zeros(np.shape(self.gNode), dtype=np.float32)
        self.gNode = np.array(self.gNode, dtype=np.float32)
        self.gElement = np.array(self.gElement, dtype=np.float32)
        [node_number, dummy] = np.shape(self.gNode)
        self.node_number = node_number
        self.f1 = np.zeros(node_number * 6, dtype=np.float32)
        self.gravity = np.zeros(node_number * 6, dtype=np.float32)
        self.gravity_copy = self.gravity.copy()
        self.l1 = self.node_number - 1
        T0 = TransformMatrix(self.gNode)
        self.k0 = StiffnessMatrix(self.gNode, self.gMaterial)
        self.gK = np.zeros((node_number * 6, node_number * 6), dtype=np.float32)
        self.gK_zero_copy = self.gK.copy()
        self.AssembleStiffnessMatrix(T0)

        self.A1 = np.zeros([0, np.dot(6, np.shape(self.gNode)[0])], dtype=np.float32)
        self.b1 = np.zeros([0, 1], dtype=np.float32)

        self.X = MatrixStruct(self.x)
        self.X.send_to_gpu()
        self.Xtriangle = MatrixStruct(self.xtriangle)
        self.Xtriangle.send_to_gpu()
        self.adjacentTriangle = np.zeros((np.shape(self.xtriangle)[0], 10), dtype=np.float32)
        SearchAdjacentTriangle = mod.get_function("SearchAdjacentTriangle")
        self.AdjacentTriangle = MatrixStruct(self.adjacentTriangle)
        self.AdjacentTriangle.send_to_gpu()
        self.Adjx = np.zeros(np.shape(self.xtriangle)[0], dtype=np.int32)
        SearchAdjacentTriangle(self.Xtriangle._cptr, self.AdjacentTriangle._cptr, cuda.InOut(self.Adjx),
                               block=(1024, 1, 1),
                               grid=(self.xtriangle.shape[0], 1, 1))
        self.adjTri = self.AdjacentTriangle.get_from_gpu(self.adjacentTriangle.shape[0])
        self.SolveModel = mod.get_function("SolveModel")
        self.TransformMatrix = mod.get_function("TransformMatrix")
        self.skeletonlist = np.zeros([100, 3], dtype=np.float32)
        self.x = np.array(self.x, dtype=np.float32)
        self.xtriangle = np.array(self.xtriangle, dtype=np.int32)
        self.vNormal, self.distance0 = fa_vector(self.x, self.xtriangle, self.skeletonlist)
        # ----------------------------------------------------------------------------------------注意
        # self.vNormal = - self.vNormal

        self.VNormal = MatrixStruct(self.vNormal)
        self.Distance0 = MatrixStruct(self.distance0)
        self.VNormal.send_to_gpu()
        self.Distance0.send_to_gpu()
        # start0 = np.array([16.7788709, -2.7065929, 5.1082195], dtype=np.float32)
        # goal0 = np.array([29.5939312, 18.0450401, 1.7955891], dtype=np.float32)
        start0 = [24.8284928, 4.0543492, -0.6314446]
        goal0 = [29.5939312, 18.0450401, 1.7955891]
        self.topnode_start = self.topnode.copy()
        # self.path0 = pathplan11(self.skeletonlist, start0, goal0)
        # start0 = 2173
        # goal0 = 0
        self.path0 = None  # pathplan11(start0, goal0)

        [bc_number, dummy] = np.shape(self.gBC1)
        # bc_number初始约束数目，2个约束只能沿着轴向移动，6个约束跟随
        self.Aeq = np.zeros((bc_number, 6 * np.shape(self.gNode)[0]), dtype=np.float32)
        self.beq = np.zeros((bc_number, 1), dtype=np.float32)

        for ibc in range(bc_number):
            n = self.gBC1[ibc][0]
            d = self.gBC1[ibc][1]
            m = np.int64((n - 1) * 6 + d - 1)
            # Aeq = np.row_stack((Aeq, np.zeros((1, 6 * np.shape(gNode)[0]))))
            self.Aeq[ibc][m] = 1
            # beq = np.row_stack((beq, np.zeros((1, 1))))
            self.beq[ibc] = self.gBC1[ibc][2]
        # for ieq in range(2):
        #     n = self.gBC1[0][0]
        #     m = np.int64((n - 1) * 6)
        #     self.Aeq[ieq + 6 + bc_number][m] = self.topnode[0, ieq + 1]-self.topnode[1, ieq + 1]
        #     self.Aeq[ieq + 6 + bc_number][m + ieq] = -(self.topnode[0, 0] - self.topnode[1, 0])
        #     self.beq[ieq + 6 + bc_number] = 0 # self.topnode[0, ieq + 1] * self.topnode[1, 0] - self.topnode[1, ieq + 1] * self.topnode[0, 0]

        self.gDelta1 = np.zeros(np.shape(self.gNode)[0] * 6, dtype=np.float32)
        self.gDelta_d = np.zeros(np.shape(self.gNode)[0] * 6, dtype=np.float32)
        self.gDelta = np.zeros(np.shape(self.gNode)[0] * 6, dtype=np.float32)
        self.F = np.zeros(np.shape(self.gNode)[0] * 6, dtype=np.float32)
        self.t = 1
        # rotatebool = 0
        self.rx, self.ry = (0, 0)
        self.tx, self.ty = (0, 0)
        self.zpos = 0
        self.rotate = self.move = False
        # gNode = gNode + 50
        self.gNode_start = self.gNode.copy()
        self.beq_start = self.beq.copy()
        self.Aeq_start = self.Aeq.copy()
        self.gK_start = self.gK.copy()
        self.gDelta_start = np.zeros(np.shape(self.gNode)[0] * 6, dtype=np.float32)
        self.gDeltadd = np.zeros(np.shape(self.gNode)[0] * 6, dtype=np.float32)
        self.ff = 0.
        self.camulatebool = True
        self.ResultNode = self.gNode.copy()
        self.ResultNodedd = self.gNode.copy()
        self.set_locate = False

        self.constraint_triangle = np.zeros((1000, 1), dtype=np.float32)
        self.Constraint_triangle = MatrixStruct(self.constraint_triangle)
        self.Constraint_triangle.send_to_gpu()
        '''c_key = np.zeros((np.shape(xtriangle)[0] + 1, 1), dtype=np.float32)
        C_key = MatrixStruct(c_key)
        C_key.send_to_gpu()'''
        self.A = np.zeros((1000, 6 * np.shape(self.gNode)[0]), dtype=np.float32)

        self.AA = MatrixStruct(self.A)
        self.AA.send_to_gpu()
        self.b = np.zeros((1000, 1), dtype=np.float32)
        self.B = MatrixStruct(self.b)
        self.B.send_to_gpu()
        self.REsultNode = MatrixStruct(self.ResultNode)
        self.REsultNode.send_to_gpu()
        self.GNode = MatrixStruct(self.gNode)
        self.GNode.send_to_gpu()
        self.a0a0 = np.zeros(1, dtype=np.int32)

        self.gDelta_d_copy = np.zeros_like(self.gDelta)
        self.rotatebool = 0

        # Ax = cuda.mem_alloc(a0a0.nbytes)
        # cuda.memcpy_htod(Ax, a0a0)
        # solver_osqp = osqp.OSQP()
        # solver_osqp.setup(P=sp.dia_matrix(gK), q=f1, A=A, l=b, u=b, **settings)
        self.na = 1
        self.la = 1
        self.sim_sheathinsert_copy = shared_udp['sheathinsert'][:,3]
        self.sheathinsert_copy = self.sim_sheathinsert_copy.copy()
        self.sim_tubeinsert_copy = 0
        # self.skel.show()
        self.forsum = 0
        self.alltime = 0
        self.Aeq_position = 0
        self.tT = np.zeros((node_number - 1, 12, 12))
        self.T = MatrixStruct(self.tT)
        self.T.send_to_gpu()
        # shared_mem['constraint_triangle'] = self.constraint_triangle.tolist()
        # shared_mem['A'] = self.A.tolist()
        # shared_mem['b'] = self.b.tolist()
        # shared_mem['gK'] = self.gK.tolist()
        self.camulatebool = False

        self.TransformMatrix(self.REsultNode._cptr, self.T._cptr, block=(12, 12, 1),
                             grid=(self.node_number - 1, 1, 1))  # cost 12ms，引入GPU后消除
        self.tT = self.T.get_from_gpu3()

        shared_draw['F'] = self.F
        shared_draw['constraint_triangle'] = self.constraint_triangle
        shared_draw['ResultNode'] = self.ResultNode
        shared_draw['l1'] = self.l1
        shared_draw['na'] = self.na
        shared_draw['la'] = self.la
        shared_draw['velocity_InsertionUnit'] = self.velocity_InsertionUnit

        self.location_copy = shared_location['location'].copy()
        cy = linalg.expm(np.cross(np.eye(3), [0., 1., 0.] / linalg.norm([0, 1, 0]) * 3.1415926535 / 2))
        camera_location = eulerAnglesToRotationMatrix(shared_location['location'][3:6], 'zyx') @ cy
        self.camera_location_copy = camera_location.copy()
        Quat = rotateToQuaternion(camera_location)
        self.Angles_copy = QuaternionToAxialAngle(Quat)
        # self.camera_location_copy_copy = self.camera_location_copy.copy()
        # self.Angles_copy_copy = self.Angles_copy.copy()
        self.d_locate = np.zeros(6, dtype=np.float32)

        rt = linalg.expm(np.cross(np.eye(3), [1., 0., 0.] / linalg.norm([1, 0, 0]) * 1.31))  # direction
        self.camera_T = self.tT[-1, 0:3, 0:3] @ rt  # self.camera_location_copy.copy() # @ rt
        self.camera_T_start = self.camera_T.copy()
        self.camera_matrix = np.array([self.ResultNode[-1][0], self.ResultNode[-1][1], self.ResultNode[-1][2],
                                       self.ResultNode[-1][0] + self.camera_T[0][0],
                                       self.ResultNode[-1][1] + self.camera_T[1][0],
                                       self.ResultNode[-1][2] + self.camera_T[2][0], self.camera_T[0][2],
                                       self.camera_T[1][2], self.camera_T[2][2]])
        if Is_warm_start:
            with open(os.getcwd() + "/data/run_data" + "/Is_warm_start.txt", "r") as f:
                for line in f:
                    data_line = line.strip("\n").split()
            with open(data_line[0], "rb") as f:
                self.ResultNode = pickle.load(f)
                self.l1 = pickle.load(f)
                self.camera_matrix = pickle.load(f)
                self.F = pickle.load(f)
                self.f1 = pickle.load(f)
                self.gDelta = pickle.load(f)
        shared_draw['camera_matrix'] = self.camera_matrix
        shared_draw['camera_T_start'] = self.camera_T_start

    def init_tube(self, shared_draw):
        self.dirs_index = 0
        self.hGimbalAngle = np.zeros(2, dtype=np.float32)
        self.hGimbalAngle_copy = np.zeros(2, dtype=np.float32)
        node_num = 100
        self.gNode, self.gElement = getnode(self.topnode, node_num, True)
        [node_number, dummy] = np.shape(self.gNode)
        self.l1 = self.node_number - 1
        T0 = TransformMatrix(self.gNode)
        self.k0 = StiffnessMatrix(self.gNode, self.gMaterial)
        self.gK = np.zeros((node_number * 6, node_number * 6), dtype=np.float32)

        self.AssembleStiffnessMatrix(T0)

        [bc_number, dummy] = np.shape(self.gBC1)
        self.Aeq = np.zeros((bc_number, 6 * np.shape(self.gNode)[0]), dtype=np.float32)
        self.beq = np.zeros((bc_number, 1), dtype=np.float32)

        for ibc in range(bc_number):
            n = self.gBC1[ibc][0]
            d = self.gBC1[ibc][1]
            m = np.int64((n - 1) * 6 + d - 1)
            self.Aeq[ibc][m] = 1
            self.beq[ibc] = self.gBC1[ibc][2]

        self.gNode_start = self.gNode.copy()
        self.beq_start = self.beq.copy()
        self.Aeq_start = self.Aeq.copy()
        self.gK_start = self.gK.copy()
        self.gDelta_start = np.zeros(np.shape(self.gNode)[0] * 6, dtype=np.float32)
        self.ff = 0.
        self.rotatebool = 0
        self.camulatebool = True
        self.ResultNode = self.gNode.copy()
        self.set_locate = False

        self.REsultNode = MatrixStruct(self.ResultNode)
        self.REsultNode.send_to_gpu()
        self.GNode = MatrixStruct(self.gNode)
        self.GNode.send_to_gpu()
        self.a0a0 = np.zeros(1, dtype=np.int32)

        self.gDelta_d_copy = np.zeros_like(self.gDelta)

        self.na = 1
        self.la = 1
        self.sim_sheathinsert_copy = self.sheathinsert_copy.copy()
        self.sim_tubeinsert_copy = 0

        self.forsum = 0
        self.alltime = 0
        self.Aeq_position = 0
        self.tT = np.zeros((node_number - 1, 12, 12))
        self.T = MatrixStruct(self.tT)
        self.T.send_to_gpu()
        self.camulatebool = False

        self.TransformMatrix(self.REsultNode._cptr, self.T._cptr, block=(12, 12, 1),
                             grid=(self.node_number - 1, 1, 1))  # cost 12ms，引入GPU后消除
        self.tT = self.T.get_from_gpu3()

        # rt = linalg.expm(np.cross(np.eye(3), [1., 0., 0.] / linalg.norm([1, 0, 0]) * 0.5))  # direction
        self.camera_T = self.tT[-1, 0:3, 0:3]  # @ rt
        self.camera_T_start = self.camera_T.copy()

        self.camera_matrix = np.array([self.ResultNode[-1][0], self.ResultNode[-1][1], self.ResultNode[-1][2],
                                       self.ResultNode[-1][0] + self.camera_T[0][0],
                                       self.ResultNode[-1][1] + self.camera_T[1][0],
                                       self.ResultNode[-1][2] + self.camera_T[2][0], self.camera_T[0][2],
                                       self.camera_T[1][2], self.camera_T[2][2]])

        shared_draw['F'] = self.F
        shared_draw['constraint_triangle'] = self.constraint_triangle
        shared_draw['ResultNode'] = self.ResultNode
        shared_draw['l1'] = self.l1
        shared_draw['na'] = self.na
        shared_draw['la'] = self.la
        shared_draw['velocity_InsertionUnit'] = self.velocity_InsertionUnit
        shared_draw['camera_matrix'] = self.camera_matrix
        shared_draw['camera_T_start'] = self.camera_T_start

    def AssembleStiffnessMatrix(self, T):
        # self.gK[:,:] = 0 #  = np.zeros((node_number * 6, node_number * 6), dtype = np.float32)
        for i in range(np.shape(self.k0)[1]):
            if self.l1 < self.node_number - 40./300.*self.node_number:
                if i < self.l1 - 18./300.*self.node_number:
                    m = 0
                elif i < self.l1:
                    m = 1
                elif i < self.node_number - 40./300.*self.node_number:
                    m = 2
                else:
                    m = 3
            else:
                if i < self.l1 - 18./300.*self.node_number:
                    m = 0
                elif i < self.l1:
                    m = 1
                else:
                    m = 3
            self.gK[6 * i:6 * i + 12, 6 * i:6 * i + 12] += np.dot(np.dot(T[i], self.k0[m, i]), np.transpose(T[i]))

    def RunFEM(self, shared_mem, shared_fps, action, velocity_sheath, shared_draw, ctx, shared_udp, shared_location):

        if self.rotatebool == 0:
            if self.camulatebool:
                # self.t += 1
                # 重新输入梁单元位置到GPU
                self.REsultNode.rehtod(self.ResultNode)
                self.a0a0[0] = 0
                # 计算碰撞约束
                # self.SolveModel(self.REsultNode._cptr, self.X._cptr, self.Xtriangle._cptr, self.REsultNode._cptr,
                #                 self.VNormal._cptr, self.Distance0._cptr,
                #                 self.Constraint_triangle._cptr, cuda.InOut(self.a0a0), self.AA._cptr, self.B._cptr,
                #                 self.AdjacentTriangle._cptr,
                #                 cuda.In(self.Adjx), block=(self.gNode.shape[0], 1, 1),
                #                 grid=(self.xtriangle.shape[0], 1, 1))
                # 计算新的旋转矩阵
                self.TransformMatrix(self.REsultNode._cptr, self.T._cptr, block=(12, 12, 1),
                                     grid=(self.node_number - 1, 1, 1))  # cost 12ms，引入GPU后消除
                # T为局部坐标转全局坐标
                self.tT = self.T.get_from_gpu3()
                # 获取新的约束
                self.constraint_triangle = self.Constraint_triangle.get_from_gpu(self.a0a0[0])
                self.A = self.AA.get_from_gpu(self.a0a0[0])
                self.b = self.B.get_from_gpu(self.a0a0[0])
                # 重新计算全局刚度矩阵
                # k = resetgK(self.k0, self.l1)
                self.gK = self.gK_zero_copy.copy()
                self.AssembleStiffnessMatrix(self.tT)

                # 求解位移变化,每个节点六个维度，前三个为位移，后三个为轴角，轴代表旋转轴，模长代表旋转角度
                self.beq[6:9] = self.beq[0:3]  # 加了两个固定点,后面等于前面的位移
                self.gDelta_d = solve_qp(sp.dia_matrix(self.gK), np.array(-self.f1 + self.F),
                                         sp.csc_matrix(self.A),
                                         self.b.flatten(),
                                         sp.csc_matrix(self.Aeq),
                                         self.beq.flatten(), solver='osqp')
                # print('F is ', np.linalg.norm(self.F))
                # self.U = np.dot(np.dot(self.gDelta_d.T, self.gK), self.gDelta_d)
                # print('U is ', self.U)
                # print('dU is ', 0.5 * self.U + np.dot(self.gDelta_d.T, -self.f1 + self.F))
                # 求解可能无解， 作一极小刚体位移
                if self.gDelta_d is None:
                    self.gDelta_d = self.gDelta_d_copy.copy() + 0.0000001
                    print('warn:QP can not be solved!')

                # for di in range(self.gDelta_d.shape[0]):
                #     if self.gDelta_d[di] > 3.1415926535:
                #         self.gDelta_d[di] -= 3.1415926535
                #     if self.gDelta_d[di] < -3.1415926535:
                #         self.gDelta_d[di] += 3.1415926535

                #     self.camera_location_copy = self.camera_location_copy_copy.copy()
                #     self.Angles_copy = self.Angles_copy_copy.copy()
                # else:
                #     self.camera_location_copy_copy = self.camera_location_copy.copy()
                #     self.Angles_copy_copy = self.Angles_copy.copy()

                # print("Creazy:", np.linalg.norm(self.gDelta_d))
                # if np.linalg.norm(self.gDelta_d)>10:
                #     self.gDelta_d=self.gDelta_d_copy.copy()

                # self.la = np.linalg.norm(self.gDelta_d)
                self.la = 1.0
                # # print(self.la)
                # if self.la > 1.0:
                #     self.gDelta_d /= (self.la * self.velocity_InsertionUnit)
                #     self.f1 = self.F + (self.f1 - self.F) / (self.la * self.velocity_InsertionUnit)
                #     self.Aeq_position += 1 / (self.la * self.velocity_InsertionUnit)
                #     cx = linalg.expm(np.cross(np.eye(3), [1., 0., 0.] / linalg.norm([1, 0, 0]) * self.gDelta_d[-3]))
                #     cy = linalg.expm(np.cross(np.eye(3), [0., 1., 0.] / linalg.norm([1, 0, 0]) * self.gDelta_d[-2]))
                #     cz = linalg.expm(np.cross(np.eye(3), [0., 0., 1.] / linalg.norm([1, 0, 0]) * self.gDelta_d[-1]))
                #     self.camera_T = np.dot(cz, np.dot(cy, np.dot(cx, self.camera_T)))
                # else:
                #     self.la = 1
                #     self.gDelta_d /= self.velocity_InsertionUnit
                #     # self.f1 = self.F + (self.f1 - self.F) / self.velocity_InsertionUnit
                #     self.Aeq_position += 1 / self.velocity_InsertionUnit
                # cx = linalg.expm(np.cross(np.eye(3), [1., 0., 0.] / linalg.norm([1, 0, 0]) * self.gDelta_d[-3]))
                # cy = linalg.expm(np.cross(np.eye(3), [0., 1., 0.] / linalg.norm([1, 0, 0]) * self.gDelta_d[-2]))
                # cz = linalg.expm(np.cross(np.eye(3), [0., 0., 1.] / linalg.norm([1, 0, 0]) * self.gDelta_d[-1]))
                # self.camera_T = np.dot(cz, np.dot(cy, np.dot(cx, self.camera_T)))
                # ct = linalg.expm(np.cross(np.eye(3), self.gDelta_d[-3:]))
                # self.camera_T = ct @ self.camera_T

                self.F += np.dot(self.gK, self.gDelta_d)
                # self.F = self.f1.copy()
                self.gDelta = self.gDelta + self.gDelta_d
                self.beq[0:3] = 0.
                ct = linalg.expm(np.cross(np.eye(3), self.gDelta[-3:]))
                self.camera_T = ct @ self.camera_T_start

            # [node_number, dummy] = np.shape(gNode)
            self.ResultNode = np.zeros(np.shape(self.gNode), dtype=np.float32)  # 这里千万不能赋值为gNode，不然会变成gNode的别名
            for i in range(self.gNode.shape[0]):
                self.ResultNode[i][0] = self.gNode[i][0] + self.gDelta[6 * i + 0] + 0.0
                self.ResultNode[i][1] = self.gNode[i][1] + self.gDelta[6 * i + 1] + 0.0
                self.ResultNode[i][2] = self.gNode[i][2] + self.gDelta[6 * i + 2] + 0.0

            self.camera_matrix = np.array([self.ResultNode[-1][0], self.ResultNode[-1][1], self.ResultNode[-1][2],
                                           self.ResultNode[-1][0] + self.camera_T[0][0],
                                           self.ResultNode[-1][1] + self.camera_T[1][0],
                                           self.ResultNode[-1][2] + self.camera_T[2][0], self.camera_T[0][2],
                                           self.camera_T[1][2], self.camera_T[2][2]])
            # print(self.camera_T)
            # self.camera_matrix = np.array([self.ResultNode[-1][0], self.ResultNode[-1][1], self.ResultNode[-1][2],
            #                                2 * self.ResultNode[-1][0] - self.ResultNode[-2][0],
            #                                2 * self.ResultNode[-1][1] - self.ResultNode[-2][1],
            #                                2 * self.ResultNode[-1][2] - self.ResultNode[-2][2], self.camera_T[0][2],
            #                                self.camera_T[1][2], self.camera_T[2][2]])
            self.shared_camera_matrix['camera_matrix'] = self.camera_matrix.copy()
            if self.Is_save_data:
                path_root = os.getcwd()
                localtime = time.localtime()
                time_now = time.strftime("/%Y-%m-%d/h%H/%H-%M-%S", localtime)
                if not os.path.exists(path_root + "/data/run_data" + time_now):
                    os.makedirs(path_root + "/data/run_data" + time_now)
                    self.dirs_index = 0
                self.data_time_stream = path_root + "/data/run_data" + time_now + "/%d.pickle" % self.dirs_index
                with open(self.data_time_stream, "wb") as f:
                    pickle.dump(self.ResultNode, f)
                    pickle.dump(self.l1, f)
                    pickle.dump(self.camera_matrix, f)
                    pickle.dump(self.F, f)
                    pickle.dump(self.f1, f)
                    pickle.dump(self.gDelta, f)
                    self.dirs_index += 1
        # DisplayResults(self.x, self.xtriangle, self.path0, self.spherelist_1, self.spherelist_2, self.F, self.rx,
        #                self.ry, self.tx, self.ty, self.zpos, self.constraint_triangle, self.ResultNode, self.l1,
        #                self.facelist, self.na, self.la, self.velocity_InsertionUnit,
        #                self.camera_matrix)  # cost 18ms

        # endtime = time.time()
        # self.alltime += (endtime - starttime)
        # self.forsum += 1
        # atime = time.time() - starttime
        # if atime < 0.005:
        #     fps1 = 0
        # else:
        #     fps1 = 1 / atime
        # shared_fps['fps'] = [fps1]

        self.rotatebool = 1
        self.camulatebool = False
        datafloat = shared_udp['datafloat']
        # print(datafloat)
        self.hGimbalAngle[0] = datafloat[0]  # 104 - 158
        self.hGimbalAngle[1] = datafloat[1]  # -79 - -132
        steermode = int(datafloat[4])  
        sim_sheathinsert = shared_udp['sheathinsert'][:, 3]
        sim_tubeinsert = datafloat[3] # 新版本里面暂时将插入部的插入距离去掉了

        dx = (sim_tubeinsert - self.sim_tubeinsert_copy)
        # dl = (sim_sheathinsert - self.sim_sheathinsert_copy)
        dl = np.dot((sim_sheathinsert - self.sim_sheathinsert_copy) * 1000, self.ResultNode[1] - self.ResultNode[0])
        self.sim_tubeinsert_copy = sim_tubeinsert
        self.sim_sheathinsert_copy = sim_sheathinsert
        if abs(dx) > abs(dl):
            len_element = np.linalg.norm(self.ResultNode[1] - self.ResultNode[0])
            mbeq = dx / len_element * (self.ResultNode[1] - self.ResultNode[0])  # 此处参数-1 可调插入部抽出速度
            self.beq[0] = mbeq[0]
            self.beq[1] = mbeq[1]
            self.beq[2] = mbeq[2]

            # if self.Aeq[-1, -1] == 0 and self.Aeq_position >= 1:
            #     for i in range(self.Aeq.shape[1] - 6):
            #         self.Aeq[0:6, -i - 1] = self.Aeq[0:6, -i - 7]
            #     self.Aeq_position -= 1
            #     self.Aeq[0:6, 0:6] = 0

            self.l1 -= dx / len_element
            if self.l1 < 0:
                self.l1 = 0
            if self.l1 > self.node_number - 1:
                self.l1 = self.node_number - 1

            # if self.l1 < self.gNode.shape[0] - 1 and dx < 0:
            #     self.na += 1
            # if self.l1 > self.gNode.shape[0] - 20 and dx > 0:
            #     self.na -= 1

            self.rotatebool = 0
            self.camulatebool = True

        elif abs(dx) < abs(dl):
            '''len_element = np.linalg.norm(self.ResultNode[1] - self.ResultNode[0])
            mbeq = dl / len_element * (self.ResultNode[1] - self.ResultNode[0])  # 参数-1调整体抽出速度
            self.beq[0] = mbeq[0]
            self.beq[1] = mbeq[1]
            self.beq[2] = mbeq[2]'''
            len_element = np.linalg.norm(self.ResultNode[1] - self.ResultNode[0])
            mbeq = dl/len_element**2 * (self.ResultNode[1] - self.ResultNode[0])  # 参数-1调整体抽出速度
            self.beq[0] = mbeq[0]
            self.beq[1] = mbeq[1]
            self.beq[2] = mbeq[2]  
            # if self.Aeq[-1, -1] == 0 and self.Aeq_position >= 1:
            #     for i in range(self.Aeq.shape[1] - 6):
            #         self.Aeq[0:6, -i - 1] = self.Aeq[0:6, -i - 7]
            #     self.Aeq_position -= 1
            #     self.Aeq[0:6, 0:6] = 0
            self.rotatebool = 0
            self.camulatebool = True
        # print(sim_sheathinsert, sim_tubeinsert, hGimbalAngle, steermode)
        tube_bend = 1
        max_velocity = 1.0
        # if sim_tubeinsert > 10:
        #     tube_bend = 0.002
        # elif sim_tubeinsert > 20:
        #     tube_bend = 0.003

        delta_hGimbalAngle = self.hGimbalAngle - self.hGimbalAngle_copy

        if delta_hGimbalAngle[0] > max_velocity:
            delta_hGimbalAngle[0] = max_velocity
            self.hGimbalAngle_copy[0] += max_velocity
        elif delta_hGimbalAngle[0] < -max_velocity:
            delta_hGimbalAngle[0] = -max_velocity
            self.hGimbalAngle_copy[0] -= max_velocity
        else:
            self.hGimbalAngle_copy[0] = self.hGimbalAngle[0]

        if delta_hGimbalAngle[1] > max_velocity:
            delta_hGimbalAngle[1] = max_velocity
            self.hGimbalAngle_copy[1] += max_velocity
        elif delta_hGimbalAngle[1] < -max_velocity:
            delta_hGimbalAngle[1] = -max_velocity
            self.hGimbalAngle_copy[1] -= max_velocity
        else:
            self.hGimbalAngle_copy[1] = self.hGimbalAngle[1]

        if steermode == 1:
            # self.f1[-3:] -= 0.001 * np.transpose(self.camera_T[0:3, 1])  # 此处参数调插入部转向速度
            self.f1[-3:] += -2 * delta_hGimbalAngle[0] * tube_bend * self.camera_T_start[0:3, 2] + 2 * \
                            delta_hGimbalAngle[
                                1] * tube_bend * self.camera_T_start[0:3, 1]  # 此处参数调插入部转向速度

            self.rotatebool = 0
            self.camulatebool = True

        if steermode == 2:
            # self.f1[-3:] -= 0.001 * np.transpose(self.camera_T[0:3, 1])  # 此处参数调插入部转向速度
            self.f1[int(self.l1) * 6 + 3:int(self.l1) * 6 + 6] += -25 * delta_hGimbalAngle[0] * 0.15 * self.camera_T[
                                                                                                        0:3,
                                                                                                        2] + \
                                                                  25 * delta_hGimbalAngle[1] * 0.15 * self.camera_T[
                                                                                                          0:3,
                                                                                                          1]  # 此处参数调鞘转向速度+

            self.rotatebool = 0
            self.camulatebool = True
        # ----------------------------------------------------  用末端点定位作为约束，达到跟随的效果
        # 注意 在四元数转轴角时使得轴角不能再超过360度，未来在转动角度超过360度时可能会出现bug
        # 注意 两个轴角之间的差不代表他们之间的旋转过程，有更复杂的表示，除非同轴才能直接相减
        # loc = shared_location['location']
        # cy = linalg.expm(np.cross(np.eye(3), [0., 1., 0.] / linalg.norm([0, 1, 0]) * 3.1415926535 / 2))
        # camera_location = eulerAnglesToRotationMatrix(loc[3:6], 'zyx') @ cy
        # camera_location_delta = camera_location @ np.transpose(self.camera_T)  #(self.camera_T)  # (self.camera_location_copy)
        # Quat = rotateToQuaternion(camera_location_delta)
        # Angles = QuaternionToAxialAngle(Quat)
        # for ieq in range(6):
        #     self.Aeq[ieq + self.gBC1.shape[0], self.node_number*6 - 6 + ieq] = 1
        #     if ieq < 3:
        #         self.beq[ieq + self.gBC1.shape[0]] = loc[ieq] - self.ResultNode[-1][ieq]  # self.location_copy[ieq]  # self.ResultNode[-1][ieq]
        #     else:
        #         self.beq[ieq + self.gBC1.shape[0]] = Angles[ieq-3]
        # self.location_copy = loc.copy()
        # self.Angles_copy = Angles.copy()
        # self.camera_location_copy = camera_location.copy()
        # self.rotatebool = 0
        # self.camulatebool = True
        # -----------------------------------------------------------------------------------

        # --------------------------------------------------用末端点与定位点之间的差作为受力输入
        #         loc = shared_location['location']
        #         cy = linalg.expm(np.cross(np.eye(3), [0., 1., 0.] / linalg.norm([0, 1, 0]) * 3.1415926535 / 2))
        #         camera_location = eulerAnglesToRotationMatrix(loc[3:6], 'zyx') @ cy
        #         camera_location_delta = camera_location @ np.transpose(self.camera_T)
        #         Quat = rotateToQuaternion(camera_location_delta)
        #         Angles = QuaternionToAxialAngle(Quat)
        #         position_locate = loc[0:3] - self.ResultNode[-1]
        #         if np.linalg.norm(position_locate) != 0 or np.linalg.norm(Angles) != 0:
        #             position_loc = position_locate @ self.camera_T
        #             len_element = np.linalg.norm(self.ResultNode[1] - self.ResultNode[0])
        #             mbeq = position_loc[0] / len_element * (self.ResultNode[1] - self.ResultNode[0])  # 参数-1调整体抽出速度
        #             self.beq[0] = mbeq[0]
        #             self.beq[1] = mbeq[1]
        #             self.beq[2] = mbeq[2]
        #             self.rotatebool = 0
        #             self.camulatebool = True
        #             self.f1[-12:] += 0.05 * np.dot(self.gK[-12:, -3:], Angles.T)
        #             self.f1[-3:] += 0.01 * position_loc[1] * self.camera_T[0:3, 2] - 0.01 * position_loc[2]* self.camera_T[0:3, 1]
        # ------------------------------------------------------------------------------------
        #         self.gravity = self.gravity_copy.copy()

        if shared_draw['event'][2] == 5:  # e.type == pygame.QUIT:
            sys.exit()
        elif shared_draw['event'][0] == 1:  # e.type == pygame.KEYDOWN:
            if shared_draw['event'][1] == 1:  # e.key == pygame.K_ESCAPE:
                sys.exit()
            if shared_draw['event'][1] == 2:  # e.key == pygame.K_1:
                self.hGimbalAngle = np.zeros(2, dtype=np.float32)
                self.hGimbalAngle_copy = np.zeros(2, dtype=np.float32)
                self.ff = 0.
                self.gNode = self.gNode_start.copy()
                self.beq = self.beq_start.copy()
                self.Aeq = self.Aeq_start.copy()
                self.gK = self.gK_start.copy()
                self.gDelta = self.gDelta_start.copy()
                self.f1 = np.zeros(self.node_number * 6, dtype=np.float32)
                self.F = np.zeros(self.node_number * 6, dtype=np.float32)
                self.rotatebool = 0
                self.camulatebool = True
                self.l1 = self.node_number - 1
                self.na = 0
                self.topnode = self.topnode_start.copy()
                self.forsum = 0
                self.alltime = 0
                self.Aeq_position = 0
                self.camera_T = self.camera_T_start.copy()

            elif not shared_draw['set_locate']:
                if shared_draw['event'][1] == 3:  # e.key == pygame.K_2:
                    shared_draw['set_locate'] = 1
                    self.set_locate = True
                    self.hGimbalAngle = np.zeros(2, dtype=np.float32)
                    self.hGimbalAngle_copy = np.zeros(2, dtype=np.float32)
                    self.ff = 0.
                    self.gNode = self.gNode_start.copy()
                    self.beq = self.beq_start.copy()
                    self.Aeq = self.Aeq_start.copy()
                    self.gK = self.gK_start.copy()
                    self.gDelta = self.gDelta_start.copy()
                    self.f1 = np.zeros(self.node_number * 6, dtype=np.float32)
                    self.F = np.zeros(self.node_number * 6, dtype=np.float32)
                    self.rotatebool = 0
                    self.camulatebool = True
                    self.l1 = self.node_number - 1
                    self.na = 0

                    self.forsum = 0
                    self.alltime = 0
                    self.Aeq_position = 0
                    self.camera_T = self.camera_T_start.copy()

                elif shared_draw['event'][1] == 4:  # e.key == pygame.K_w:
                    # 如果怎么调都有维度不对，上面有刚度可以调self.gMaterial
                    velocity_insert = 0.05
                    mbeq = -velocity_insert * (self.ResultNode[1] - self.ResultNode[0])  # 此处参数-1 可调插入部抽出速度
                    self.beq[0] = mbeq[0]
                    self.beq[1] = mbeq[1]
                    self.beq[2] = mbeq[2]
                    if self.Aeq[-1, -1] == 0 and self.Aeq_position >= 1:
                        for i in range(self.Aeq.shape[1] - 6):
                            self.Aeq[0:6, -i - 1] = self.Aeq[0:6, -i - 7]
                        self.Aeq_position -= 1
                        self.Aeq[0:6, 0:6] = 0
                    # if self.l1 < self.gNode.shape[0] - 1:
                    #     self.na += 1
                    #     if self.na > (
                    #             19 + self.la * self.velocity_InsertionUnit):  # (self.la * self.velocity_InsertionUnit):  # 此处调显示时的对应速度
                    #         self.l1 += 1
                    #         self.na = 0
                    self.l1 += velocity_insert
                    if self.l1 < 0:
                        self.l1 = 0
                    if self.l1 > self.node_number - 1:
                        self.l1 = self.node_number - 1

                    self.rotatebool = 0
                    self.camulatebool = True

                elif shared_draw['event'][1] == 5:  # e.key == pygame.K_s:
                    velocity_insert = 0.05
                    mbeq = velocity_insert * (self.ResultNode[1] - self.ResultNode[0])  # 此处参数1 可调插入部插入速度
                    self.beq[0] = mbeq[0]
                    self.beq[1] = mbeq[1]
                    self.beq[2] = mbeq[2]
                    if self.Aeq[0, 0] == 0 and self.Aeq_position >= 1:
                        for i in range(self.Aeq.shape[1] - 6):
                            self.Aeq[0:6, i] = self.Aeq[0:6, i + 6]
                        self.Aeq_position -= 1
                        self.Aeq[0:6, -6:] = 0
                    # if self.l1 > 5:
                    #     self.na -= 1
                    #     if self.na < 0:
                    #         self.l1 -= 1
                    #         self.na = (
                    #                     19 + self.la * self.velocity_InsertionUnit)  # (self.la * self.velocity_InsertionUnit) 此处调显示时的对应速度, 上面还有一个display里要调的
                    self.l1 -= velocity_insert
                    if self.l1 < 0:
                        self.l1 = 0
                    if self.l1 > self.node_number - 1:
                        self.l1 = self.node_number - 1

                    self.rotatebool = 0
                    self.t1 = True
                    self.camulatebool = True

                elif shared_draw['event'][1] == 6:  # e.key == pygame.K_z:
                    self.f1[-3:] -= 0.1 * np.transpose(self.camera_T[0:3, 0])  # 此处参数调插入部转向速度
                    self.rotatebool = 0
                    self.camulatebool = True

                elif shared_draw['event'][1] == 7:  # e.key == pygame.K_x:
                    xx = linalg.expm(np.cross(np.eye(3), [1., 0., 0.] / linalg.norm([1, 0, 0]) * 0.01))
                    self.camera_T = self.camera_T @ xx
                    self.camera_T_start = self.camera_T.copy()
                    self.rotatebool = 0
                    self.camulatebool = True

                elif shared_draw['event'][1] == 8:  # e.key == pygame.K_a:
                    # self.f1[-3:] -= 0.2 * 0.001 * np.transpose(self.camera_T[0:3, 1])  # 此处参数调插入部转向速度
                    self.f1[-3:] -= 0.2 * np.transpose(self.camera_T_start[0:3, 1])

                    self.rotatebool = 0
                    self.camulatebool = True

                elif shared_draw['event'][1] == 9:  # e.key == pygame.K_d:
                    # self.f1[-3:] += 0.2 * 0.001 * np.transpose(self.camera_T[0:3, 1])  # 此处参数调插入部转向速度
                    self.f1[-3:] += 0.2 * np.transpose(self.camera_T_start[0:3, 1])
                    self.rotatebool = 0
                    self.camulatebool = True

                elif shared_draw['event'][1] == 10:  # e.key == pygame.K_q:
                    self.f1[-3:] += 0.2 * np.transpose(self.camera_T_start[0:3, 2])  # 此处参数调鞘转向速度
                    self.rotatebool = 0
                    self.camulatebool = True
                elif shared_draw['event'][1] == 11:  # e.key == pygame.K_e:
                    self.f1[-3:] -= 0.2 * np.transpose(self.camera_T_start[0:3, 2])  # 此处参数调鞘转向速度
                    self.rotatebool = 0
                    self.camulatebool = True
                elif shared_draw['event'][1] == 12:  # e.key == pygame.K_UP:
                    mbeq = -0.05 * (self.ResultNode[1] - self.ResultNode[0])  # 参数-1调整体抽出速度
                    self.beq[0] = mbeq[0]
                    self.beq[1] = mbeq[1]
                    self.beq[2] = mbeq[2]
                    # if self.Aeq[-1, -1] == 0 and self.Aeq_position >= 1:
                    #     for i in range(self.Aeq.shape[1] - 6):
                    #         self.Aeq[0:6, -i - 1] = self.Aeq[0:6, -i - 7]
                    #     self.Aeq_position -= 1
                    #     self.Aeq[0:6, 0:6] = 0
                    self.rotatebool = 0
                    self.camulatebool = True

                elif shared_draw['event'][1] == 13:  # e.key == pygame.K_DOWN:
                    mbeq = 0.05 * (self.ResultNode[1] - self.ResultNode[0])  # 参数1调整体插入速度
                    self.beq[0] = mbeq[0]
                    self.beq[1] = mbeq[1]
                    self.beq[2] = mbeq[2]
                    # if self.Aeq[0, 0] == 0 and self.Aeq_position >= 1:
                    #     for i in range(self.Aeq.shape[1] - 6):
                    #         self.Aeq[0:6, i] = self.Aeq[0:6, i + 6]
                    #     self.Aeq_position -= 1
                    #     self.Aeq[0:6, -6:] = 0

                    self.rotatebool = 0
                    self.camulatebool = True

                elif shared_draw['event'][1] == 14:  # e.key == pygame.K_LEFT:
                    self.f1[int(self.l1) * 6 + 3:int(self.l1) * 6 + 6] -= 1.2 * np.transpose(
                        self.camera_T[0:3, 2])  # 此处参数调鞘转向速度
                    self.rotatebool = 0
                    self.camulatebool = True

                elif shared_draw['event'][1] == 15:  # e.key == pygame.K_RIGHT:
                    self.f1[int(self.l1) * 6 + 3:int(self.l1) * 6 + 6] += 1.2 * np.transpose(
                        self.camera_T[0:3, 2])  # 此处参数调鞘
                    self.rotatebool = 0
                    self.camulatebool = True

                elif shared_draw['event'][1] == 16:  # e.key == pygame.K_o:
                    self.f1[int(self.l1) * 6 + 3:int(self.l1) * 6 + 6] += 1.2 * np.transpose(
                        self.camera_T[0:3, 1])  # 此处参数调鞘转向速度
                    self.rotatebool = 0
                    self.camulatebool = True

                elif shared_draw['event'][1] == 17:  # e.key == pygame.K_i:
                    self.f1[int(self.l1) * 6 + 3:int(self.l1) * 6 + 6] -= 1.2 * np.transpose(
                        self.camera_T[0:3, 1])  # 此处参数调鞘转向速度
                    self.rotatebool = 0
                    self.camulatebool = True

                elif shared_draw['event'][1] == 18:  # e.key == pygame.K_SPACE:
                    # self.gravity += set_gravity(self.node_number, self.gMaterial[0, 3], self.gMaterial[0, 2], 9.8, 7,
                    #                            self.tT, self.gravity)
                    # print(self.gravity)
                    self.rotatebool = 0
                    self.camulatebool = True

                elif shared_draw['event'][1] == 32:  # e.key == pygame.K_r:
                    # self.gravity = self.gravity_copy.copy()
                    self.rotatebool = 0
                    self.camulatebool = True
                elif shared_draw['event'][1] == 34:  # e.key == pygame.K_4:
                    # 直接进行上下左右移动，注意只在无碰撞情况使用，否则可能出现一些bug
                    mbeq = 0.05 * np.array([np.linalg.norm(self.ResultNode[1] - self.ResultNode[0]), 0, 0])
                    self.beq[0] = mbeq[0]
                    self.beq[1] = mbeq[1]
                    self.beq[2] = mbeq[2]
                    self.rotatebool = 0
                    self.camulatebool = True


                elif shared_draw['event'][1] == 35:  # e.key == pygame.K_5:
                    # 直接进行上下左右移动，注意只在无碰撞情况使用，否则可能出现一些bug
                    mbeq = -0.05 * np.array([np.linalg.norm(self.ResultNode[1] - self.ResultNode[0]), 0, 0])
                    self.beq[0] = mbeq[0]
                    self.beq[1] = mbeq[1]
                    self.beq[2] = mbeq[2]
                    self.rotatebool = 0
                    self.camulatebool = True
                elif shared_draw['event'][1] == 36:  # e.key == pygame.K_6:
                    # 直接进行上下左右移动，注意只在无碰撞情况使用，否则可能出现一些bug
                    mbeq = 0.05 * np.array([0, np.linalg.norm(self.ResultNode[1] - self.ResultNode[0]), 0])
                    self.beq[0] = mbeq[0]
                    self.beq[1] = mbeq[1]
                    self.beq[2] = mbeq[2]
                    self.rotatebool = 0
                    self.camulatebool = True
                elif shared_draw['event'][1] == 37:  # e.key == pygame.K_7:
                    # 直接进行上下左右移动，注意只在无碰撞情况使用，否则可能出现一些bug
                    mbeq = -0.05 * np.array([0, np.linalg.norm(self.ResultNode[1] - self.ResultNode[0]), 0])
                    self.beq[0] = mbeq[0]
                    self.beq[1] = mbeq[1]
                    self.beq[2] = mbeq[2]
                    self.rotatebool = 0
                    self.camulatebool = True
                elif shared_draw['event'][1] == 38:  # e.key == pygame.K_8:
                    # 直接进行上下左右移动，注意只在无碰撞情况使用，否则可能出现一些bug
                    mbeq = 0.05 * np.array([0, 0, np.linalg.norm(self.ResultNode[1] - self.ResultNode[0])])
                    self.beq[0] = mbeq[0]
                    self.beq[1] = mbeq[1]
                    self.beq[2] = mbeq[2]
                    self.rotatebool = 0
                    self.camulatebool = True
                elif shared_draw['event'][1] == 39:  # e.key == pygame.K_9:
                    # 直接进行上下左右移动，注意只在无碰撞情况使用，否则可能出现一些bug
                    mbeq = -0.05 * np.array([0, 0, np.linalg.norm(self.ResultNode[1] - self.ResultNode[0])])
                    self.beq[0] = mbeq[0]
                    self.beq[1] = mbeq[1]
                    self.beq[2] = mbeq[2]
                    self.rotatebool = 0
                    self.camulatebool = True

            elif shared_draw['set_locate']:  # reset the node locate
                if shared_draw['event'][1] == 19:  # e.key == pygame.K_3:
                    shared_draw['set_locate'] = 0
                    self.set_locate = False
                    print(self.topnode)
                    self.init_tube(shared_draw)

                elif shared_draw['event'][1] == 20:  # e.key == pygame.K_w:
                    self.topnode[0, 0] += 1
                    self.gNode, self.gElement = getnode(self.topnode, 100, True)
                    self.rotatebool = 0
                    self.camulatebool = False

                elif shared_draw['event'][1] == 21:  # e.key == pygame.K_s:
                    self.topnode[0, 0] -= 1
                    self.gNode, self.gElement = getnode(self.topnode, 100, True)
                    self.rotatebool = 0
                    self.camulatebool = False

                elif shared_draw['event'][1] == 22:  # e.key == pygame.K_a:
                    self.topnode[0, 2] -= 1
                    self.gNode, self.gElement = getnode(self.topnode, 100, True)
                    self.rotatebool = 0
                    self.camulatebool = False

                elif shared_draw['event'][1] == 23:  # e.key == pygame.K_d:
                    self.topnode[0, 2] += 1
                    self.gNode, self.gElement = getnode(self.topnode, 100, True)
                    self.rotatebool = 0
                    self.camulatebool = False

                elif shared_draw['event'][1] == 24:  # e.key == pygame.K_q:
                    self.topnode[0, 1] -= 1
                    self.gNode, self.gElement = getnode(self.topnode, 100, True)
                    self.rotatebool = 0
                    self.camulatebool = False

                elif shared_draw['event'][1] == 25:  # e.key == pygame.K_e:
                    self.topnode[0, 1] += 1
                    self.gNode, self.gElement = getnode(self.topnode, 100, True)
                    self.rotatebool = 0
                    self.camulatebool = False

                elif shared_draw['event'][1] == 26:  # e.key == pygame.K_i:
                    self.topnode[1, 0] += 1
                    self.gNode, self.gElement = getnode(self.topnode, 100, True)
                    self.rotatebool = 0
                    self.camulatebool = False

                elif shared_draw['event'][1] == 27:  # e.key == pygame.K_k:
                    self.topnode[1, 0] -= 1
                    self.gNode, self.gElement = getnode(self.topnode, 100, True)
                    self.rotatebool = 0
                    self.camulatebool = False

                elif shared_draw['event'][1] == 28:  # e.key == pygame.K_j:
                    self.topnode[1, 1] -= 1
                    self.gNode, self.gElement = getnode(self.topnode, 100, True)
                    self.rotatebool = 0
                    self.camulatebool = False

                elif shared_draw['event'][1] == 29:  # e.key == pygame.K_l:
                    self.topnode[1, 1] += 1
                    self.gNode, self.gElement = getnode(self.topnode, 100, True)
                    self.rotatebool = 0
                    self.camulatebool = False

                elif shared_draw['event'][1] == 30:  # e.key == pygame.K_u:
                    self.topnode[1, 2] -= 1
                    self.gNode, self.gElement = getnode(self.topnode, 100, True)
                    self.rotatebool = 0
                    self.camulatebool = False

                elif shared_draw['event'][1] == 31:  # e.key == pygame.K_o:
                    self.topnode[1, 2] += 1
                    self.gNode, self.gElement = getnode(self.topnode, 100, True)
                    self.rotatebool = 0
                    self.camulatebool = False
                elif shared_draw['event'][1] == 33:  # e.key == pygame.K_0:
                    path_root = os.getcwd()
                    newnode = np.load("/home/p9/Projects/embodiedrobot/navigation/data/data/transed_points.npy")
                    #self.topnode = np.array(newnode * 1000)
                    self.topnode[1] = newnode[0] * 1000 + 10*(newnode[0]-newnode[1])/np.linalg.norm(newnode[0]-newnode[1])
                    self.topnode[0] = self.topnode[1] + 300*(newnode[0]-newnode[1])/np.linalg.norm(newnode[0]-newnode[1])
                    self.gNode, self.gElement = getnode(self.topnode, 100, True)
                    self.rotatebool = 0
                    self.camulatebool = False

        shared_draw['event'] = np.array([0, 0, shared_draw['event'][2]])
        shared_draw['F'] = self.F
        shared_draw['constraint_triangle'] = self.constraint_triangle
        shared_draw['ResultNode'] = self.ResultNode
        shared_draw['l1'] = self.l1
        shared_draw['na'] = self.na
        shared_draw['la'] = self.la
        shared_draw['camera_matrix'] = self.camera_matrix

        return self.ResultNode, self.camera_matrix


def fem(shared_mem, shared_fps, action, velocity_InsertionUnit, velocity_sheath, shared_camera_matrix, trans_matrix,
        tip_matrix, shared_draw, shared_udp, mode, shared_location, Is_warm_start):
    cuda.init()
    ctx = cuda.Device(0).make_context()
    mod = get_cuda_SourceModule()
    try:
        model = FEM(shared_mem, velocity_InsertionUnit, shared_camera_matrix, mod, trans_matrix, tip_matrix,
                    shared_draw, mode, shared_location, Is_warm_start, shared_udp)
        while 1:
            State_s, camera_matrix = model.RunFEM(shared_mem, shared_fps, action, velocity_sheath, shared_draw, ctx,
                                                  shared_udp, shared_location)
    except SystemExit:
        print('FEM正常退出')
    except (Exception, BaseException) as errorstring:
        print('FEM退出，原因是：%s' % str(errorstring))
        print(traceback.format_exc())
        path_root = os.getcwd()
        with open(path_root + "/data/run_data" + "/Is_warm_start.txt", "w") as f:
            f.write(model.data_time_stream)
    finally:
        ctx.pop()
        event = np.array([0, 0, 5])
        shared_draw['event'] = event
