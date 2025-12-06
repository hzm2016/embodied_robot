# from pickle import FALSE
# import glfw as glfw
import time
# import freetype
import copy
from multiprocessing import Lock, Manager, Process, shared_memory

import cv2 as cv
import numpy as np

# from AR_moudle import AR
# from AuxiliaryDisplay_moudle import \
#     AuxiliaryDisplay
# from Motion_moudle import Motion
# from STLModel_moudle import STLModel
# from ObjModel_moudle import ObjModel
# from WorldCamera_moudle import WorldCamera

from ar.src.AR_moudle import AR
from ar.src.AuxiliaryDisplay_moudle import \
    AuxiliaryDisplay
from ar.src.Motion_moudle import Motion
from ar.src.STLModel_moudle import STLModel
from ar.src.ObjModel_moudle import ObjModel
from ar.src.WorldCamera_moudle import WorldCamera

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *


class Project:
    Independent = False
    num = 1
    IF_AR = True
    If_AR_Real = True
    If_combine_in_one_screen = True
    Width_List = [1540, 1107, 640, 700, 2214, 2514, 1700, 558]
    Height_List = [845, 623, 480, 700, 623, 623, 700, 500]

    Width1 = Width_List[3]
    Height1 = Height_List[3]

    if num == 1:
        Width = Width1
        Height = Height1
    elif num == 2:
        Width = 2 * Width1
        Height = Height1       


    
    centerpoint_x = Width / 2
    centerpoint_y = Height / 2
    screenrate_x = centerpoint_x + 8
    screenrate_y = centerpoint_y + 31

    m_scale = 1
    m_MouseDownPT = [Width1 / 2, Height1 / 2]

    m_bMouseDown = False

    speed_ws = 1.2
    speed_ad = 0.3
    speed_ud = 0.4
    angle_change = [0, 0, 0]
    angle_with_x = angle_change[0]
    angle_with_y = angle_change[1]
    angle_with_z = angle_change[2]


    fov = 60.0

    ticks = 0
    lastTime = 0

    fpp_on = False
    tpp_on = False

    # keyboard state
    U = False
    Q = False
    O = False
    P = False
    J = False
    K = False
    L = False
    N = False
    M = False
    B = False

    def __init__(self, shared_camera_matrix):

        self.shared_camera_matrix = shared_camera_matrix
        self.m_position = np.array((-34.85 + 119.5154705, -52.8 + 138.55568695, -19.93 + 105.82824516))
        self.m_orientation = np.array((0.683, 0.644, 0.341))
        self.m_orientation_delta = np.array((0.0, 0.0, 0.0))
        self.m_up = np.array((0.0, 1.0, 0.0))
        self.m_left = np.cross(self.m_up, self.m_orientation)
        self.m_up = np.cross(self.m_orientation, self.m_left)

        self.TPP = None

        self.keypoints = np.zeros((5, 3))
        self.Lookatparam = [
            self.m_position[0], self.m_position[1], self.m_position[2],
            self.m_orientation[0] + self.m_position[0],
            self.m_orientation[1] + self.m_position[1],
            self.m_orientation[2] + self.m_position[2],
            self.m_up[0], self.m_up[1], self.m_up[2]
        ]
        # self.Lookatparam = [0,0,0,0,0,1,0,1,0]

        #################### 入口图 ######################
        # self.m_position = np.array((-34.85, -52.8, -19.93))
        # self.m_orientation = np.array((0.683 ,0.644, 0.341))
        # self.m_up = np.array((0.0,1.0,0.0))
        ####################################################

        #################### 侧视图 ######################
        # self.m_position = np.array((-54.88611607,  -36.96915122, -185.77141157))
        # self.m_orientation = np.array((0.29889257, 0.21649844, 0.92755202))
        # self.m_up = np.array((-0.74086449 , 0.5812467 ,  0.33646813))
        ####################################################
        self.WorldCamera = WorldCamera()
        self.motion = Motion()
        self.NewImage = np.zeros(self.Width1 * self.Height1 * 3)
        # self.NewImage = np.zeros((self.Width1 , self.Height1 , 3))
        # self.angle = self.motion.Included_Angle(self.m_orientation)

    def Model_Init(self):
        global model, model_, Axis3D, Keypoint, KeySphere, FixedSphere, PerspectiveLine, PerspectiveLine1, PerspectiveLine2
         # read path
        with open('/home/p9/Projects/embodiedrobot/navigation/data/makehole.txt') as f:
            folderpath = f.readline().replace('\r', '').replace('\n', '')
            print("-------------------------------------")
            print(folderpath)

        objPath = folderpath + '/stl/HEAD2_hole.stl'
        
        model = STLModel(folderpath + '/stl/BRAIN2_hole.stl', 1)
        model_ = STLModel(folderpath + '/stl/VEN2_hole.stl', 1)
        # model  = STLModel(r"G:\first_person_perspective\res\3DMODELS\BRAIN_hole_smooth2.stl",1)
        # model_ = STLModel(r"G:\first_person_perspective\res\3DMODELS\MRI-models\MRI-target.stl",1)

        # model = ObjModel(r"G:\first_person_perspective\res\3DMODELS\OBJ\VesselModel_LPS.obj")
        # model_ = ObjModel(r"G:\first_person_perspective\res\3DMODELS\OBJ\VesselModel_LPS.obj")
        # model = STLModel("res/3DMODELS/BRAIN_hole_smooth2.stl", 1)
        # model_= STLModel("res/3DMODELS/MRI-models/MRI-target.stl", 1)
        Auxiliary_model = AuxiliaryDisplay()
        # Axis3D = Auxiliary_model.DrawCylinder(self.m_position, self.angle, 8)
        self.Transfer = self.motion.matrix_to_Transfer(-self.m_left.T, self.m_up.T, -self.m_orientation.T)
        Axis3D = Auxiliary_model.DrawCylinder(self.m_position, self.Transfer, 8)
        Keypoint = Auxiliary_model.DrawPoint(1, np.array((0, 0, 0)))
        KeySphere = Auxiliary_model.DrawSphere(0.2, 30, 30)
        FixedSphere = Auxiliary_model.DrawSphere(0.2, 30, 30)
        PerspectiveLine = Auxiliary_model.DrawPerspectiveLine(self.m_position, self.m_orientation, self.m_up)
        PerspectiveLine1 = Auxiliary_model.DrawPerspectiveLine(self.m_position, self.m_orientation, self.m_up)
        PerspectiveLine2 = Auxiliary_model.DrawPerspectiveLine(self.m_position, self.m_orientation, self.m_up)
        # DistanceLine = Auxiliary_model.DrawLine(self.m_position, self.m_position)
        # DistanceLine1 = Auxiliary_model.DrawLine(self.m_position, self.m_position)

    def display_init(self):

        model.Vertex = model.Vertex @ np.transpose(self.shared_camera_matrix['trans_matrix'][0:3, 0:3])
        model.Vertex += 1000 * np.transpose(self.shared_camera_matrix['trans_matrix'][0:3, 3])
        model.Draw_Init()
        # model_.MidPos += np.array((59.41809383, -28.68455701, -95.25279284))
        # # model_.MidPos[0,:] = [ 99.4942347 , 124.76675584, 131.4105491 ]
        # model_.Vertex += np.array((59.41809383, -28.68455701, -95.25279284))

        ############################################################
        model_.MidPos = model_.MidPos @ np.transpose(self.shared_camera_matrix['trans_matrix'][0:3, 0:3])
        model_.Vertex = model_.Vertex @ np.transpose(self.shared_camera_matrix['trans_matrix'][0:3, 0:3])
        model_.MidPos += 1000 * np.transpose(self.shared_camera_matrix['trans_matrix'][0:3, 3])
        model_.Vertex += 1000 * np.transpose(self.shared_camera_matrix['trans_matrix'][0:3, 3])
        model_.Draw_Init()

        Axis3D.Draw_Init()

        KeySphere.Draw_Init()
        FixedSphere.Draw_Init()

    def AR_init(self):
        global ar

        ar = AR(self.FPP.Screen[0], self.FPP.Screen[1])
        if self.If_AR_Real:
            path = '0'
            real = True
        else:
            path = r"G:\first_person_perspective\res\back0.png"
            real = False
        ar.CV_Init(path, real)

    def Perspective_Init(self):
        # 设置FPP(First Person Perspective)的glulookat
        FPP_lookat = [
            self.m_position[0], self.m_position[1], self.m_position[2],
            self.m_orientation[0] + self.m_position[0],
            self.m_orientation[1] + self.m_position[1],
            self.m_orientation[2] + self.m_position[2],
            self.m_up[0], self.m_up[1], self.m_up[2]
        ]
        view_window = [0, 0, self.Width1, self.Height1]
        screen = [self.Width1, self.Height1]
        self.FPP = Perspective(view_window, screen, FPP_lookat)
        self.FPP.If_Draw[1] = False
        self.FPP.ReverseBackColor = True
        self.fpp_on = True
        self.FPP.fov = self.fov

        # 定义光源属性变量*/
        self.FPP.LightSource.LightNum = 0
        self.FPP.LightSource.DoubleFace = True
        self.FPP.LightSource.position = [self.m_position[0], self.m_position[1], self.m_position[2], 1]
        self.FPP.LightSource.orientation = [self.m_orientation[0], self.m_orientation[1], self.m_orientation[2]]
        self.FPP.LightSource.cutoff = [45]
        self.FPP.LightSource.exponent = [35]
        self.FPP.LightSource.diffuse = [1, 1, 1, 1]
        self.FPP.LightSource.K = [1, 0.007, 0]

        if self.num > 1:
            # 设置TPP(Third Person Perspective)的glulookat
            TPP_lookat = [
                self.WorldCamera.position[0], self.WorldCamera.position[1], self.WorldCamera.position[2],
                self.WorldCamera.orientation[0] + self.WorldCamera.position[0],
                self.WorldCamera.orientation[1] + self.WorldCamera.position[1],
                self.WorldCamera.orientation[2] + self.WorldCamera.position[2],
                self.WorldCamera.up[0], self.WorldCamera.up[1], self.WorldCamera.up[2]
            ]

            view_window = [self.Width1, 0, self.Width - self.Width1, self.Height - self.Height1]
            screen = [self.Width - self.Width1, self.Height - self.Height1]
            self.TPP = Perspective(view_window, screen, TPP_lookat)
            self.tpp_on = True

            self.TPP.If_Draw = [True, True, True, False, False, False, False, False, False]
            self.TPP.Blend_State = True
            # 定义光源属性变量*/
            self.TPP.LightSource.LightNum = 1
            self.TPP.LightSource.DoubleFace = True
            self.TPP.LightSource.position = [self.WorldCamera.position[0], self.WorldCamera.position[1],
                                             self.WorldCamera.position[2], 1]
            self.TPP.LightSource.orientation = [self.WorldCamera.orientation[0], self.WorldCamera.orientation[1],
                                                self.WorldCamera.orientation[2]]
            self.TPP.LightSource.cutoff = [40]
            self.TPP.LightSource.exponent = [1]
            self.TPP.LightSource.diffuse = [1, 1, 1, 1]
            self.TPP.LightSource.K = [1.5, 0, 0]

    def Perspective_Param_Update(self):
        # 设置FPP(First Person Perspective)的glulookat
        if self.Independent:
            FPP_lookat = [
                self.m_position[0], self.m_position[1], self.m_position[2],
                self.m_orientation[0] + self.m_position[0],
                self.m_orientation[1] + self.m_position[1],
                self.m_orientation[2] + self.m_position[2],
                self.m_up[0], self.m_up[1], self.m_up[2]
            ]
            self.Lookatparam = FPP_lookat
            self.FPP.lookatparam = self.Lookatparam
        else:
            self.FPP.lookatparam = self.Lookatparam  #务必直接引用self.Lookatparam, self.Lookatparam可能与修正后的posi, oti, up不同
        # print(self.FPP.lookatparam)
        self.FPP.View_Window = [0, 0, self.Width1, self.Height1]
        self.FPP.Screen = [self.Width1, self.Height1]
        self.FPP.If_Draw[2] = False
        self.FPP.If_Draw[4] = KeySphere.If_Draw
        self.FPP.If_Draw[5] = PerspectiveLine.If_Draw
        self.FPP.If_Draw[6] = PerspectiveLine1.If_Draw
        self.FPP.If_Draw[7] = PerspectiveLine2.If_Draw
        self.FPP.If_Draw[8] = FixedSphere.If_Draw
        # 定义光源属性变量*/
        if self.Independent:
            self.FPP.LightSource.position = [self.m_position[0], self.m_position[1], self.m_position[2], 1]
            self.FPP.LightSource.orientation = [self.m_orientation[0], self.m_orientation[1], self.m_orientation[2]]
        else:
            self.FPP.LightSource.position = [self.Lookatparam[0], self.Lookatparam[1], self.Lookatparam[2], 1]
            self.FPP.LightSource.orientation = [self.Lookatparam[3] - self.Lookatparam[0],
                                                self.Lookatparam[4] - self.Lookatparam[1],
                                                self.Lookatparam[5] - self.Lookatparam[2]]
        # K0 = [0.48]#常数衰减因子
        # K1 = [0.03]#线性衰减因子
        # K2 = [0.00001]#二次衰减因子
        if self.num > 1:
            # 设置TPP(Third Person Perspective)的glulookat
            TPP_lookat = [
                self.WorldCamera.position[0], self.WorldCamera.position[1], self.WorldCamera.position[2],
                self.WorldCamera.orientation[0] + self.WorldCamera.position[0],
                self.WorldCamera.orientation[1] + self.WorldCamera.position[1],
                self.WorldCamera.orientation[2] + self.WorldCamera.position[2],
                self.WorldCamera.up[0], self.WorldCamera.up[1], self.WorldCamera.up[2]
            ]
            try:
                self.TPP.View_Window = [self.Width1, 0, self.Width - self.Width1, self.Height - self.Height1]
            except:
                view_window = [self.Width1, 0, self.Width - self.Width1, self.Height - self.Height1]
                screen = [self.Width - self.Width1, self.Height - self.Height1]
                self.TPP = Perspective(view_window, screen, TPP_lookat)
                self.fpp_on = True
                self.TPP.View_Window = [self.Width1, 0, self.Width - self.Width1, self.Height - self.Height1]
                self.TPP.If_Draw = [True, True, True, False, False, False, False, False, False]
                self.TPP.Blend_State = True
                # 定义光源属性变量*/
                self.TPP.LightSource.LightNum = 1
                self.TPP.LightSource.DoubleFace = True
                self.TPP.LightSource.position = [self.WorldCamera.position[0], self.WorldCamera.position[1],
                                                 self.WorldCamera.position[2], 1]
                self.TPP.LightSource.orientation = [self.WorldCamera.orientation[0], self.WorldCamera.orientation[1],
                                                    self.WorldCamera.orientation[2]]
                self.TPP.LightSource.cutoff = [40]
                self.TPP.LightSource.exponent = [1]
                self.TPP.LightSource.diffuse = [1, 1, 1, 1]
                self.TPP.LightSource.K = [1.5, 0, 0]
            self.TPP.Screen = [self.Width - self.Width1, self.Height - self.Height1]
            self.TPP.lookatparam = TPP_lookat
            Axis3D.position = self.m_position
            self.Transfer = self.motion.matrix_to_Transfer(-self.m_left.T, self.m_up.T, -self.m_orientation.T)
            Axis3D.Transfer = self.Transfer
            self.TPP.Depth_State = False
            # 定义光源属性变量*/
            self.TPP.LightSource.position = [self.WorldCamera.position[0], self.WorldCamera.position[1],
                                             self.WorldCamera.position[2], 1]
            self.TPP.LightSource.orientation = [self.WorldCamera.orientation[0], self.WorldCamera.orientation[1],
                                                self.WorldCamera.orientation[2]]

    def tick_check(self):
        if self.ticks == 0:
            self.lastTime = time.time()
        self.ticks += 1
        if (self.ticks == 120):
            deltaTime = time.time()
            print("帧率：" + str(self.ticks / (deltaTime - self.lastTime)))
            self.ticks = 0

    def print_info(self):
        print("\nPosition\n")
        print(self.m_position)
        print("\nOrientation\n")
        print(self.m_orientation)
        print("\nUp\n")
        print(self.m_up)


    def AuxiliaryLine_Display_Control(self, world_posi, idx):
        if self.L:
            if idx == 1:
                PerspectiveLine1.Draw_Init()  # 初始化绘制射线
                PerspectiveLine1.If_Draw = True
            elif idx == 2:
                PerspectiveLine2.Draw_Init()  # 初始化绘制射线
                PerspectiveLine2.If_Draw = True
        else:
            if idx == 1:
                PerspectiveLine1.If_Draw = False
            elif idx == 2:
                PerspectiveLine2.If_Draw = False
        if self.K:
            # DistanceLine1.posi0 = self.m_position + 2 * (self.m_position - model_.MidPos[0, :])
            # DistanceLine1.posi1 = model_.MidPos[0, :]
            # DistanceLine1.Draw_Init()
            # DistanceLine1.If_Draw = True
            win_posi_result = self.Get_World_to_Screen_Info(world_posi)
            ar.Win_Posi[0] = win_posi_result[0:2] / win_posi_result[2]

        else:
            ar.Win_Posi[0] = [int(self.Width1 / 2), int(self.Height / 2)]

    def UpdateKeySphereInfo(self):
        PerspectiveLine2.position = self.m_position
        PerspectiveLine2.orientation = self.m_orientation
        PerspectiveLine2.up = self.m_up
        PerspectiveLine2.fixed_point = KeySphere.position
        PerspectiveLine2.Find_intersection()
        # Depth = np.linalg.norm(PerspectiveLine2.Intersection - PerspectiveLine2.fixed_point)
        ar.datalist_for_show[2] = round(np.linalg.norm(KeySphere.position - self.m_position),2)
        ar.datalist_for_show[3] = np.around(PerspectiveLine2.Distance,2)
        self.AuxiliaryLine_Display_Control(KeySphere.position,2)


    def Auxiliary_Display(self):
        if self.U:
            FixedSphere.position = model_.MidPos[0, :]
            # FixedSphere.position = [ 99.4942347 , 124.76675584, 131.4105491 ]
            FixedSphere.If_Draw = True
        else:
            FixedSphere.If_Draw = False
        if self.Q:  # Q键被按下
            mouse_posi_info = self.Get_Mouse_to_World_Info(self.Qx, self.Qy)
            print(mouse_posi_info[0])
            self.keypoints[0, :] = mouse_posi_info[0]
            # self.keypoints[0, :] = [89.21429888, 90.15998842, 84.22348341]
            KeySphere.position = self.keypoints[0, :]
            #KeySphere.position = model_.MidPos[0]
            #KeySphere.position = self.m_position
            KeySphere.If_Draw = True
            #PerspectiveLine1.If_Draw = False
            # self.L = False
            # self.K = False
            self.Q = False
        if KeySphere.If_Draw:
            self.UpdateKeySphereInfo()
        else:
            ar.datalist_for_show[2] = "Null"
            ar.datalist_for_show[3] = "Null"
        if self.O:
            Distance = np.linalg.norm(model_.MidPos[0, :] - self.m_position)
            PerspectiveLine1.position = self.m_position
            PerspectiveLine1.orientation = self.m_orientation
            PerspectiveLine1.up = self.m_up
            PerspectiveLine1.fixed_point = model_.MidPos[0, :]

            PerspectiveLine1.Find_intersection()
            # Depth = np.linalg.norm(PerspectiveLine1.Intersection - PerspectiveLine1.fixed_point)
            #PerspectiveLine1.length = Depth
            # Depth = PerspectiveLine1.Distance[1]
            ar.datalist_for_show[0] = round(Distance,2)
            ar.datalist_for_show[1] = np.around(PerspectiveLine1.Distance,2)
            if not KeySphere.If_Draw:
                self.AuxiliaryLine_Display_Control(model_.MidPos[0, :], 1)
        else:
            ar.datalist_for_show[0:2] = ["Null", "Null"]
            PerspectiveLine1.If_Draw = False
            PerspectiveLine2.If_Draw = False
            ar.Win_Posi[0] = [int(self.Width1/2), int(self.Height/2)]
            # print(distance)
        if self.P:
            PerspectiveLine.position = self.m_position
            PerspectiveLine.orientation = self.m_orientation
            PerspectiveLine.up = self.m_up
            PerspectiveLine.fixed_point = model_.MidPos[0, :]
            PerspectiveLine.Find_intersection()
            PerspectiveLine.If_Draw = True
            PerspectiveLine.Draw_Init()
        else:
            PerspectiveLine.If_Draw = False


        ar.If_ShowData = True if self.N else False
        ar.If_ShowLine = True if self.M else False
        ar.If_ShowAxis = True if self.B else False


    def adjust(self):
        if np.matmul(self.m_up, self.m_orientation.T) != 0:
            self.m_left = np.cross(self.m_up, self.m_orientation)
            self.m_up = np.cross(self.m_orientation, self.m_left)


    def display(self):
        # print(self.shared_camera_matrix['camera_matrix'])
        if not self.Independent:
            self.Lookatparam = self.shared_camera_matrix['camera_matrix']
            self.m_position = np.array(self.Lookatparam[0:3])
            self.m_orientation = np.array(self.Lookatparam[3:6]) - self.m_position
            self.m_up = np.array(self.Lookatparam[6:9])
            self.adjust()
        self.Perspective_Param_Update()

        if self.If_combine_in_one_screen:
            self.FPP.DisplayPerspective()
            self.Auxiliary_Display()
            if self.FPP.Grab_State:
                # glWindowPos2d(self.Width - self.Width1, 0)
                glWindowPos2d(0, 0)
                self.NewImage = ar.newImage.flatten()
                glDrawPixels(self.Width1, self.Height1, GL_BGR, GL_UNSIGNED_BYTE,
                                np.ascontiguousarray(self.NewImage.data))
                #######glDrawPixels(Width, Height, format, type) Width 与 Height 为绘制区域的宽与高
            glutSwapBuffers()  # 交互前后缓冲区
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)  # 擦除buffer中颜色和深度缓存
        else:
            for self.scene in range(self.num):
                if self.scene == 0:
                    self.FPP.DisplayPerspective()
                    if self.num == 1:
                        self.Auxiliary_Display()
                        glutSwapBuffers()  # 交互前后缓冲区
                        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)  # 擦除buffer中颜色和深度缓存
                if self.scene == 1:

                    self.Auxiliary_Display()
                    if self.FPP.Grab_State:
                        glWindowPos2d(self.Width - self.Width1, 0)
                        # glWindowPos2d(0, 0)
                        self.NewImage = ar.newImage.flatten()
                        glDrawPixels(self.Width1, self.Height1, GL_BGR, GL_UNSIGNED_BYTE,
                                    np.ascontiguousarray(self.NewImage.data))
                        #######glDrawPixels(Width, Height, format, type) Width 与 Height 为绘制区域的宽与高
                    glutSwapBuffers()  # 交互前后缓冲区
                    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)  # 擦除buffer中颜色和深度缓存

        # self.tick_check()

    def ReshapeEvent(self, width, height):
        glViewport(0, 0, width, height)  # 视口在屏幕的大小位置
        glMatrixMode(GL_PROJECTION)  # 投影矩阵
        glLoadIdentity()  # 矩阵单位
        gluPerspective(45.0, (width * 1.0 / height), 0.1, 1000.0)  # 设置投影矩阵
        glMatrixMode(GL_MODELVIEW)  # 模型矩阵
        glLoadIdentity()  # 单位矩阵化
        glEnable(GL_DEPTH_TEST)  # 启动深度检测

    def IdleEvent(self):
        glutPostRedisplay()


    def Get_World_to_Screen_Info(self, posi):
        current_viewport = glGetIntegerv(GL_VIEWPORT)
        current_projection = glGetDoublev(GL_PROJECTION_MATRIX)
        current_modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
        Win_posi = np.array(gluProject(*posi, current_modelview, current_projection, current_viewport))
        return Win_posi


    def Get_Mouse_to_World_Info(self, x, y):

        current_viewport = glGetIntegerv(GL_VIEWPORT)
        current_projection = glGetDoublev(GL_PROJECTION_MATRIX)
        current_modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
        ScreenZ = glReadPixels(x, current_viewport[3] - y, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT)
        [objx, objy, objz] = gluUnProject(x, current_viewport[3] - y, ScreenZ, current_modelview, current_projection,
                                          current_viewport)
        # print([objx, objy, objz])
        Depth = 1.0 / ((1 / self.FPP.far - 1 / self.FPP.near) * ScreenZ + 1 / self.FPP.near)

        # print(Depth)
        obj_posi = np.array((objx, objy, objz))
        Distance = np.linalg.norm(obj_posi - self.m_position)
        # print(Distance)
        return [obj_posi, Depth, Distance]
        # print(ScreenZ)

    def KeyboardEvent(self, key, x, y):
        self.m_left = np.cross(self.m_up, self.m_orientation)
        # Movement Control
        if key == b'W':
            self.m_position += self.m_orientation * self.speed_ws
        elif key == b'S':
            self.m_position += -self.m_orientation * self.speed_ws
        elif key == b'A':
            self.m_position += self.m_left * self.speed_ad
        elif key == b'D':
            self.m_position += -self.m_left * self.speed_ad
        elif key == b' ':
            self.m_position += self.m_up * self.speed_ud
        elif key == b'C':
            self.m_position += -self.m_up * self.speed_ud
        elif key == b'F' or key == b'G':
            if key == b'F':
                self.rotate_o = 1 / 180 * np.pi
                self.angle_with_z -= self.rotate_o
            if key == b'G':
                self.rotate_o = -1 / 180 * np.pi
                self.angle_with_z -= self.rotate_o
            if self.m_orientation.shape[0] == 3:
                self.m_orientation = self.m_orientation[np.newaxis, :]
            if self.m_left.shape[0] == 3:
                self.m_left = self.m_left[np.newaxis, :]
            if self.m_up.shape[0] == 3:
                self.m_up = self.m_up[np.newaxis, :]
            self.vector_olu = np.concatenate((self.m_orientation, self.m_left, self.m_up), axis=0)
            self.vector_olu = np.matmul(self.vector_olu,
                                        np.array(self.motion.Rotate(self.m_orientation[0, :], self.rotate_o)).T)
            self.m_orientation = self.vector_olu[0, :]
            self.m_left = self.vector_olu[1, :]
            self.m_up = self.vector_olu[2, :]

        # Light Control
        elif key == b'E':
            if self.FPP.LightSource.cutoff[0] > 1:
                self.FPP.LightSource.cutoff[0] += -1
        elif key == b'R':
            self.FPP.LightSource.cutoff[0] += 1
        elif key == b'T':
            if self.FPP.LightSource.exponent[0] < 128:
                self.FPP.LightSource.exponent[0] += 1
        elif key == b'Y':
            if self.FPP.LightSource.exponent[0] > 0:
                self.FPP.LightSource.exponent[0] -= 1

        # Blend Control
        elif key == b'Z':
            self.FPP.Depth_State = True if self.FPP.Depth_State == False else False
        elif key == b'X':
            self.FPP.Blend_State = True if self.FPP.Blend_State == False else False
            if not self.FPP.Blend_State:
                self.FPP.LightSource.K = self.FPP.last[0]
                self.FPP.LightSource.exponent = self.FPP.last[1]
            else:
                self.FPP.last = [self.FPP.LightSource.K, self.FPP.LightSource.exponent]
                self.FPP.LightSource.K = [0.8, 0, 0]
                self.FPP.LightSource.exponent = [1]

        # Cam Control
        elif key == b'H':
            self.FPP.Grab_State = True if self.FPP.Grab_State == False else False

        # GL Display Control
        elif key == b'1':
            self.FPP.If_Draw[1] = 0 if self.FPP.If_Draw[1] == 1 else 1
        elif key == b'2':
            self.FPP.If_Draw[0] = 0 if self.FPP.If_Draw[0] == 1 else 1
        elif key == b'3':
            self.FPP.ReverseBackColor = True if self.FPP.ReverseBackColor == False else False
            if self.tpp_on:
                self.TPP.ReverseBackColor = True if self.TPP.ReverseBackColor == False else False
        elif key == b'6':
            self.num = 1 if self.num == 2 else 2
            self.Width1 = self.Width_List[1] if self.Width1 != self.Width_List[1] else self.Width_List[5]
            self.Height1 = self.Height_List[1] if self.Height1 != self.Height_List[1] else self.Height_List[5]

        # Auxiliary Display Control
        elif key == b'Q':
            self.Qx = x
            self.Qy = y
            # 记录Q键被按下时的，鼠标坐标
            self.Q = True
        elif key == b'I':
            Keypoint.If_Draw = False
            KeySphere.If_Draw = False
            # PerspectiveLine.If_Draw = False
            self.O = False
            self.P = False
            self.Q = False
            self.K = False
            self.J = False
            ar.datalist_for_show[2] = "Null"
        elif key == b'U':
            self.U = True if self.U == False else False
        elif key == b'O':
            self.O = True if self.O == False else False
        elif key == b'P':
            self.P = True if self.P == False else False
        elif key == b'L':
            self.L = True if self.L == False else False
        elif key == b'K':
            self.K = True if self.K == False else False
        elif key == b'J':
            self.J = True if self.L == False else False
        elif key == b'N':
            self.N = True if self.N == False else False
        elif key == b'M':
            self.M = True if self.M == False else False
        elif key == b'B':
            self.B = False if self.B else True


        if self.TPP is not None:
            if key == b'j':
                self.TPP.Depth_State = True if self.TPP.Depth_State == False else False
            elif key == b'k':
                self.TPP.Blend_State = True if self.TPP.Blend_State == False else False
            elif key == b'2':
                self.TPP.If_Draw[1] = 0 if self.TPP.If_Draw[1] == 1 else 1

    def processSpecialKeys(self, key, x, y):
        if key == GLUT_KEY_DOWN:
            model_.MidPos -= self.m_orientation * self.speed_ws
        elif key == GLUT_KEY_UP:
            model_.MidPos += self.m_orientation * self.speed_ws
        elif key == GLUT_KEY_LEFT:
            model_.MidPos += self.m_left * self.speed_ad
        elif key == GLUT_KEY_RIGHT:
            model_.MidPos -= self.m_left * self.speed_ad
        elif key == GLUT_KEY_PAGE_UP:
            model_.MidPos += self.m_up * self.speed_ud
        elif key == GLUT_KEY_PAGE_DOWN:
            model_.MidPos -= self.m_up * self.speed_ud

    def MouseEvent(self, button, state, x, y):

        if (state == GLUT_DOWN and button == GLUT_LEFT_BUTTON):

            self.m_bMouseDown = True  # 鼠标左键按下
            self.m_MouseDownPT[0] = x  # 记录当前X坐标
            self.m_MouseDownPT[1] = y  ##记录当前Y坐标
        else:
            self.m_bMouseDown = False

    def MotionEvent(self, x, y):

        if self.m_bMouseDown:

            self.offsetx = x - self.m_MouseDownPT[0]
            self.offsety = y - self.m_MouseDownPT[1]
            self.screenrate_x = self.offsetx / self.centerpoint_x * np.pi / 5
            self.angle_with_y -= self.screenrate_x
            # 用于摄像机水平移动
            self.screenrate_y = self.offsety / self.centerpoint_y * np.pi / 10
            self.angle_with_x -= self.screenrate_y
            # 用于摄像机上下移动
            self.m_left = np.cross(self.m_up, self.m_orientation)

            if self.m_orientation.shape[0] == 3:
                self.m_orientation = self.m_orientation[np.newaxis, :]
            if self.m_left.shape[0] == 3:
                self.m_left = self.m_left[np.newaxis, :]
            if self.m_up.shape[0] == 3:
                self.m_up = self.m_up[np.newaxis, :]
            self.vector_olu = np.concatenate((self.m_orientation, self.m_left, self.m_up), axis=0)
            self.vector_olu = np.matmul(self.vector_olu,
                                        np.array(self.motion.Rotate(self.m_up[0, :], self.screenrate_x)).T)
            self.vector_olu = np.matmul(self.vector_olu,
                                        np.array(self.motion.Rotate(self.vector_olu[1, :], -self.screenrate_y)).T)
            self.m_orientation = self.vector_olu[0, :]
            self.m_left = self.vector_olu[1, :]
            self.m_up = self.vector_olu[2, :]
            # print(self.vector_olu)
            self.m_MouseDownPT[0] = x  # 记录当前X坐标
            self.m_MouseDownPT[1] = y  ##记录当前Y坐标

    # 检测鼠标进入或离开窗口
    def MouseEntry(self, state):
        # print("mouse Enter")
        i = state

    def onMouseDownAndMove(self, x, y):
        # print("onMouseDownAndMove")
        i = x

    def MenuEvent(self, choose):
        if choose == 1:
            # 复位：把旋转平移缩放的值复位
            # 用于平移，对应X Y Z 平移量。按键W:上  S:下   A:左  D：右
            self.m_position = np.array((-34.85 + 119.5154705, -52.8 + 138.55568695, -19.93 + 105.82824516))

            # 用于旋转，分别是绕X轴 和Y轴旋转的角度，用鼠标左键控制
            self.m_orientation = np.array((0.683, 0.644, 0.341))

            self.m_up = np.array((0, 1, 0))

            # 用于缩放，用鼠标中间滚轮控制 [弃用]
            self.m_scale = 1.0

            # 记录鼠标坐标点，用于控制旋转角度；
            self.m_MouseDownPT[0] = self.Width / 2
            self.m_MouseDownPT[1] = self.Height / 2

            # 记录鼠标左键是否按下，按下为true,初始值为false
            self.m_bMouseDown = False
        elif choose == 2:
            self.m_position = np.array(
                (-54.88611607 + 119.5154705, -36.96915122 + 138.55568695, -185.77141157 + 105.82824516))
            self.m_orientation = np.array((0.29889257, 0.21649844, 0.92755202))
            self.m_up = np.array((-0.74086449, 0.5812467, 0.33646813))

    def Loop(self):
        while True:
            glutDisplayFunc(self.display)  # 重新绘制事件
            glutKeyboardFunc(self.KeyboardEvent)  # 键盘事件
            glutMouseFunc(self.MouseEvent)  # 鼠标事件
            glutMotionFunc(self.MotionEvent)
            # glutPassiveMotionFunc(self.MotionEvent)
            # glutReshapeFunc(self.ReshapeEvent)   #窗口大小发生变化事件
            # glutMotionFunc(self.onMouseDownAndMove)     #鼠标移动事件
            glutIdleFunc(self.IdleEvent)  # 空闲处理事件
            # glutEntryFunc(self.MouseEntry)       #检测鼠标进入或离开窗口
            glutCreateMenu(self.MenuEvent)  # 创建菜单
            glutAddMenuEntry("reset", 1)  # 菜单项1
            glutAddMenuEntry("left", 2)  # 菜单项2
            glutAttachMenu(GLUT_RIGHT_BUTTON)  # 鼠标右键按下弹出菜单
            glutTimerFunc(10, self.Loop, 1)

    def main(self):
        glutInit()  # 初始化glut
        self.Model_Init()
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)  # 设置模式为：双缓冲，深度缓存区
        glutInitWindowPosition(0, 0)  # 窗口位置
        glutInitWindowSize(self.Width, self.Height)  # 窗口大小
        # glutInitWindowSize(1107, 623) #窗口大小
        glutCreateWindow("project1")  # 创建窗口
        self.display_init()

        self.Perspective_Init()
        if self.IF_AR:
            self.AR_init()
        # self.Loop()
        glutDisplayFunc(self.display)  # 重新绘制事件
        glutKeyboardFunc(self.KeyboardEvent)  # 键盘事件
        glutSpecialFunc(self.processSpecialKeys)
        glutMouseFunc(self.MouseEvent)  # 鼠标事件
        glutMotionFunc(self.MotionEvent)
        # glutPassiveMotionFunc(self.MotionEvent)
        # glutReshapeFunc(self.ReshapeEvent)   #窗口大小发生变化事件
        # glutMotionFunc(self.onMouseDownAndMove)     #鼠标移动事件
        glutIdleFunc(self.IdleEvent)  # 空闲处理事件
        # glutEntryFunc(self.MouseEntry)       #检测鼠标进入或离开窗口
        glutCreateMenu(self.MenuEvent)  # 创建菜单
        glutAddMenuEntry("reset", 1)  # 菜单项1
        glutAddMenuEntry("left", 2)  # 菜单项2
        glutAttachMenu(GLUT_RIGHT_BUTTON)  # 鼠标右键按下弹出菜单
        glutMainLoop()  # 调用已注册的回调函数

        # print(3)


class Perspective():
    def __init__(self, view_window, screen, lookatparam):
        self.View_Window = view_window
        self.Screen = screen
        # target, brain, axis, keypoint, keysphere, PerspectiveLine, PerspectiveLine1, PerspectiveLine2
        self.If_Draw = [True, True, False, False, False, False, False, False, False]
        self.Blend_State = False
        self.Depth_State = True
        self.Grab_State = False
        self.ReverseBackColor = False
        self.BackColor = [0, 0, 0, 1]
        self.lookatparam = lookatparam
        self.LightSource = self.Light()
        self.near = 0.1
        self.far = 1000.0
        self.fov = 60.0

    class Light:

        def __init__(self):
            self.position = np.zeros(3)
            self.orientation = np.zeros(3)
            self.cutoff = [60]
            self.diffuse = np.array((1, 1, 1, 1))
            self.exponent = [1]
            self.K = [1, 0, 0]
            self.LightNum = 0
            self.DoubleFace = False

        def GenerateLight(self):
            glFrontFace(GL_CCW)  # 默认为CCW正方向
            glShadeModel(GL_SMOOTH)  # 设置着色模式为：平滑底纹

            GL_LightNum = GL_LIGHT0 + self.LightNum
            for i in range(7):
                if i != self.LightNum:
                    glDisable(GL_LIGHT0 + i)  # 关闭i号灯
                else:
                    glEnable(GL_LightNum)  # 启动LightNum号灯

            # 设置光源属性*/
            glLightfv(GL_LightNum, GL_POSITION, self.position)
            glLightfv(GL_LightNum, GL_SPOT_DIRECTION, self.orientation)
            glLightfv(GL_LightNum, GL_SPOT_CUTOFF, self.cutoff)
            glLightfv(GL_LightNum, GL_SPOT_EXPONENT, self.exponent)
            glLightfv(GL_LightNum, GL_DIFFUSE, self.diffuse)
            glLightfv(GL_LightNum, GL_CONSTANT_ATTENUATION, self.K[0])
            glLightfv(GL_LightNum, GL_LINEAR_ATTENUATION, self.K[1])
            glLightfv(GL_LightNum, GL_QUADRATIC_ATTENUATION, self.K[2])

            # 设置材质属性*/
            glEnable(GL_COLOR_MATERIAL)  # 使用颜色材质
            if self.DoubleFace:
                glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
                glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE)  # 启用双面光照
                glFrontFace(GL_CW)  # 默认为CCW正方向，此举将法向量反向，以便渲染双面。
            else:
                glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE)
            # 指定顶点的颜色属性，正反面均设置为材质的环境颜色和散射颜色，且颜色相同。
            glEnable(GL_LIGHTING)

    def DisplayPerspective(self):

        glViewport(*self.View_Window)  # 视口在屏幕的大小位置
        glMatrixMode(GL_PROJECTION)  # 投影矩阵
        glLoadIdentity()  # 矩阵单位
        gluPerspective(self.fov, (self.Screen[0] * 1.0 / self.Screen[1]), self.near, self.far)  # 设置投影矩阵
        glEnable(GL_DEPTH_TEST)  # 启动深度检测
        # -----------------------设置光照-------------------------#
        self.LightSource.GenerateLight()
        # -------------------------------------------------------#
        gluLookAt(*self.lookatparam)
        # self.print_info() #输出摄像头位置与朝向，以便观察。

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()  # 单位矩阵化
        # 设置摄像头(观察者角度)的位置，朝向以及上方向。
        # std::cout << "......." << std::endl
        # 以脑部模型中心坐标为标准进行统一位移(不能删)*/

        # model_.MidPos = model.MidPos

        # glColor4f(0.0, 0.0, 0.7, 1.0)
        Axis3D.Draw(self.If_Draw[2])

        
        if not self.If_Draw[8]:
            glColor4f(46.0 / 255, 139.0 / 255, 87.0 / 255, 1)  # 设置血肿颜色
            model_.Draw(self.If_Draw[0], 0)  # 绘制血肿
        # print(model_.offset)
        # model_->InitVBO()
        if (self.Blend_State):
            glEnable(GL_BLEND)  # 启用混合
        else:
            glDisable(GL_BLEND)
        # model->GetTriangleNum()
        # glDrawArrays(GL_TRIANGLES, 0, Triangle_num)

        # GL_SRC_ALPHA + (GL_DST_COLOR  ||  GL_SRC_COLOR) 均可，未知原因*/
        # 建议不使用ONE_MINUS，会出现亮斑*/
        glPushMatrix()
        if (self.Depth_State):
            glDepthMask(True)
            # glBlendFunc(GL_SRC_ALPHA, GL_DST_COLOR)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_DST_COLOR)
            glColor4f(250.0 / 255, 250.0 / 255, 240.0 / 255, 0.3)  # 设置脑部颜色
            # glTranslatef(-model.MidPos[0,0], -model.MidPos[0,1], -model.MidPos[0,2])
            model.Draw(self.If_Draw[1], 1)  # 绘制脑部

        else:
            glDepthMask(False)
            # glBlendFunc(GL_SRC_ALPHA, GL_DST_COLOR)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_DST_COLOR)
            glColor4f(250.0 / 255, 250.0 / 255, 240.0 / 255, 0.3)  # 设置脑部颜色
            # glTranslatef(-model.MidPos[0,0], -model.MidPos[0,1], -model.MidPos[0,2])
            model.Draw(self.If_Draw[1], 1)  # 绘制脑部
            glDepthMask(True)
        glPopMatrix()

        if self.If_Draw[3]:
            # glDisable(GL_LIGHTING)
            Keypoint.Draw()

        if self.If_Draw[4]:
            # glDisable(GL_LIGHTING)
            KeySphere.Draw()


        if self.If_Draw[8]:
            # self.Depth_State = True
            # self.Blend_State = True
            FixedSphere.Draw()
            if self.Depth_State:
                glDepthMask(True)
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_DST_COLOR)
                glColor4f(46.0 / 255, 139.0 / 255, 87.0 / 255, 1)  # 设置血肿颜色
                model_.Draw(self.If_Draw[0], 0)  # 绘制血肿
            else:
                glDepthMask(False)
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_DST_COLOR)
                glColor4f(46.0 / 255, 139.0 / 255, 87.0 / 255, 1)  # 设置血肿颜色
                model_.Draw(self.If_Draw[0], 0)  # 绘制血肿
                glDepthMask(True)

        if ar.If_ShowLine:
            glDisable(GL_LIGHTING)
            if self.If_Draw[5]:
                PerspectiveLine.Draw()

            if self.If_Draw[6]:
                PerspectiveLine1.Draw()

            if self.If_Draw[7]:
                PerspectiveLine2.Draw()


        if self.ReverseBackColor:
            glClearColor(1, 1, 1, 0)
        else:
            glClearColor(0, 0, 0, 0)

        # 获取窗口图像
        if self.Grab_State:
            ar.Grab_to_CV()

            # NewImage = ar.newImage.flatten()


# if '__name__'=='__main__':
def AR_show(shared_camera_matrix):
    # 通过在循环内读取shared_camera_matrix['camera_matrix']使用该参数，如300行左右的print
    project = Project(shared_camera_matrix)
    project.main()


if __name__ == "__main__":
    # 整合时改main函数要删掉，并改一下113行左右的路径
    # cv.namedWindow("1")
    shared_camera_matrix = dict({})
    # shared_camera_matrix['camera_matrix'] = [-34.85, -52.8, -19.93, -34.85 + 0.683, -52.8 + 0.644, -19.93 + 0.341,
    #                                          -0.439852, 0.58277, -0.219604]
    
    shared_camera_matrix['camera_matrix'] = [8.8, 8.7, 8.7, 8.91313095, 8.81313095, 8.75656586, 0.707106769, -0.500187541, -7.05335457]
    shared_camera_matrix['trans_matrix'] = np.array(([1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]))

    AR_show(shared_camera_matrix)
