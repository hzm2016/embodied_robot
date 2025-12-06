from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import numpy as np
import cv2 as cv

from ar.src.AR_moudle import AR
from ar.src.AuxiliaryDisplay_moudle import \
    AuxiliaryDisplay
from ar.src.Motion_moudle import Motion
from ar.src.STLModel_moudle import STLModel
from ar.src.ObjModel_moudle import ObjModel
from ar.src.WorldCamera_moudle import WorldCamera




class Perspective():
    def __init__(self, view_window, screen, lookatparam):
        self.View_Window = view_window
        self.Screen = screen
        self.If_Draw = [True, True]
        self.Blend_State = False
        self.Depth_State = True
        self.Grab_State = False
        self.BackColor = [0, 0, 0, 1]
        self.lookatparam = lookatparam

        
        


    class Light:

        def __init__(self):
            self.position = np.zeros(3)
            self.orientation = np.zeros(3)
            self.cutoff = [60]
            self.diffuse = np.zeros(4)
            self.exponent = [1]
            self.K = [1, 0, 0]
            self.LightNum = 0
            self.DoubleFace = False

        def GenerateLight(self):
            glFrontFace(GL_CCW)#默认为CCW正方向
            glShadeModel(GL_SMOOTH)#设置着色模式为：平滑底纹
            
            GL_LightNum = GL_LIGHT0 + self.LightNum
            for i in range (7):
                if i == self.LightNum:
                    glEnable(GL_LightNum)  #启动LightNum号灯
                else:
                    glDisable(GL_LIGHT0 + i)                #关闭i号灯
                    
            #设置光源属性*/
            glLightfv(GL_LightNum, GL_POSITION, self.position)
            glLightfv(GL_LightNum, GL_SPOT_DIRECTION, self.orientation)
            glLightfv(GL_LightNum, GL_SPOT_CUTOFF, self.cutoff)
            glLightfv(GL_LightNum, GL_SPOT_EXPONENT, self.exponent)
            glLightfv(GL_LightNum, GL_DIFFUSE, self.diffuse)
            glLightfv(GL_LightNum, GL_CONSTANT_ATTENUATION, self.K[0])
            glLightfv(GL_LightNum, GL_LINEAR_ATTENUATION, self.K[1])
            glLightfv(GL_LightNum, GL_QUADRATIC_ATTENUATION, self.K[2])

            #设置材质属性*/
            glEnable(GL_COLOR_MATERIAL)#使用颜色材质
            if self.DoubleFace:
                glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
                glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE)#启用双面光照
                glFrontFace(GL_CW)#默认为CCW正方向，此举将法向量反向，以便渲染双面。
            else:
                glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE)
            #指定顶点的颜色属性，正反面均设置为材质的环境颜色和散射颜色，且颜色相同。

    def Light_Init(self):
        self.LightSource = self.Light()



    def DisplayPerspective(self):
        
        
        glViewport(*self.View_Window)#视口在屏幕的大小位置 
        glMatrixMode(GL_PROJECTION)# 投影矩阵
        glLoadIdentity()           # 矩阵单位
        gluPerspective(45.0, (self.Screen[0]*1.0 / self.Screen[1]), 0.1, 1000.0)# 设置投影矩阵
        glEnable(GL_DEPTH_TEST)#启动深度检测
        #-----------------------设置光照-------------------------#
        self.LightSource.GenerateLight()
        #-------------------------------------------------------#
        gluLookAt(*self.lookatparam)
        #self.print_info() #输出摄像头位置与朝向，以便观察。
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()#单位矩阵化  
        #设置摄像头(观察者角度)的位置，朝向以及上方向。
        #std::cout << "......." << std::endl
        #以脑部模型中心坐标为标准进行统一位移(不能删)*/
        
        self.model_.MidPos = self.model.MidPos
        
        
        if self.If_Draw[1] == 0:
            glClearColor(1, 1, 1, 0)
        else:
            glClearColor(0 ,0 ,0, 0)

        glColor4f(46.0 / 255, 139.0 / 255, 87.0 / 255, 1) #设置血肿颜色
        
        
        self.model_.Draw(self.If_Draw[0], 0) #绘制血肿
        #model_->InitVBO()
        if (self.Blend_State):
            glEnable(GL_BLEND) #启用混合
        else:
            glDisable(GL_BLEND)
        #model->GetTriangleNum()
        #glDrawArrays(GL_TRIANGLES, 0, Triangle_num)

        #GL_SRC_ALPHA + (GL_DST_COLOR  ||  GL_SRC_COLOR) 均可，未知原因*/
        # 建议不使用ONE_MINUS，会出现亮斑*/
        
        if (self.Depth_State):
            glDepthMask(True)
            glBlendFunc(GL_SRC_ALPHA, GL_DST_COLOR)
            glColor4f(250.0/255, 250.0/255, 240.0/255, 0.3) #设置脑部颜色
            self.model.Draw(self.If_Draw[1], 1) #绘制脑部

        else:
            glDepthMask(False)
            glBlendFunc(GL_SRC_ALPHA, GL_DST_COLOR)
            glColor4f(250.0 / 255, 250.0 / 255, 240.0 / 255, 0.3) #设置脑部颜色
            self.model.Draw(self.If_Draw[1], 1) #绘制脑部
            glDepthMask(True)

        # 获取窗口图像
        if self.Grab_State:
            self.ar.Grab_to_CV()



    def AR_init(self):
        self.ar = AR(self.Width, self.Height)
        path = "res/back.png"
        self.ar.CV_Init(path)

    
