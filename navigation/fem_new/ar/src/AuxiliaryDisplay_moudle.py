from OpenGL.GLUT import*
from OpenGL.GL import *
from OpenGL.GLU import *
#import glm as glm
import numpy as np
import struct
#import glfw as glfw

class AuxiliaryDisplay:
    def __init__(self):
        pass

    class DrawSphere:
        def __init__(self, radius, slicez, slicey):
            self.radius = radius
            self.slicez = slicez
            self.slicey = slicey
            self.alpha_y_delta = 1 * np.pi / slicey
            self.alpha_z_delta = 2 * np.pi / slicez
            self.position = np.zeros(3)
            self.If_Draw = False
            

        def CalculateVertex(self, i, j):
            x = self.radius * np.sin(self.alpha_y_delta * i) * np.sin(self.alpha_z_delta * j)
            y = self.radius * np.cos(self.alpha_y_delta * i)
            z = self.radius * np.sin(self.alpha_y_delta * i) * np.cos(self.alpha_z_delta * j)
            return [x,y,z]
            


        def Draw_Init(self):
            self.index = glGenLists(1)
            #glEnable(GL_NORMALIZE)
            glNewList(self.index, GL_COMPILE)
            glBegin(GL_TRIANGLES)
            for j in range(self.slicez):
                glColor4f(1.0,0.0,0.0,1.0)
                glVertex3f(0, self.radius, 0)
                glVertex3f(*self.CalculateVertex(1, j))
                glVertex3f(*self.CalculateVertex(1, j+1))       
            glEnd()
            glBegin(GL_QUADS)
            for i in range(1, self.slicey-1, 1):
                for j in range(self.slicez):
                    glColor4f(1.0,0.0,0.0,1.0)
                    glVertex3f(*self.CalculateVertex(i,j))
                    glVertex3f(*self.CalculateVertex(i+1,j))
                    glVertex3f(*self.CalculateVertex(i+1,j+1))
                    glVertex3f(*self.CalculateVertex(i,j+1))
            glEnd()
            glBegin(GL_TRIANGLES)
            for j in range(self.slicez):
                glColor4f(1.0,0.0,0.0,1.0)
                glVertex3f(0, -self.radius, 0)
                glVertex3f(*self.CalculateVertex(self.slicey-1, j))
                glVertex3f(*self.CalculateVertex(self.slicey-1, j+1))       
            glEnd()
            glEndList()

        def Draw(self):
            glPushMatrix()
            glColor4f(1.0,0.0,0.0,1.0)
            glTranslate(*self.position)
            glCallList(self.index)
            glPopMatrix()


    class DrawPoint:
        def __init__(self, num, position):
            self.num = num
            self.position = position
            #self.color = color
            self.If_Draw = False
        
        def Draw_Init(self):
            self.ID = glGenLists(1)
            glNewList(self.ID, GL_COMPILE)
            
            glPointSize(20.0)
            glBegin(GL_POINTS)
            for i in range(self.num):
                glColor4f(1, 0, 0, 1)
                glVertex3f(*self.position[i,:])
            glEnd()
            glEndList()


        def Draw(self):
            glCallList(self.ID)


    class DrawCylinder:
        def __init__(self, position, Transfer, length):
            self.position = position
            self.Transfer = Transfer
            self.length = length

        
        def Draw_Init(self):
            # single axis specs
            bodyLen = self.length
            tipLen = self.length * 0.09
            base_radius = self.length * 0.1
            slices = 8
            stacks = 8
            # build one axis
            self.axisId = glGenLists(1)
            quadric = gluNewQuadric()
            glNewList(self.axisId, GL_COMPILE)
            gluCylinder(quadric, 0.0, base_radius, bodyLen, slices, stacks)
            glPushMatrix()
            glTranslate(0, 0, bodyLen)
            gluCylinder(quadric, base_radius, 0, tipLen, slices, stacks)
            glPopMatrix()
            glEndList()
            # now assemble all three
            self.axisIndicatorId = glGenLists(1)
            glNewList(self.axisIndicatorId, GL_COMPILE)
            # always lit and smooth shaded with a bit a spec
            glEnable(GL_LIGHTING)
            glPolygonMode(GL_FRONT, GL_FILL)
            glShadeModel(GL_SMOOTH)
            glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR,
                            [1.0, 1.0, 1.0, 1.0])
            glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, 64)
            # X axis, red
            glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE,
                            [0.7, 0.0, 0.0, 1.0])
            glColor4f(0.7, 0.0, 0.0, 1.0)
            glPushMatrix()
            
            glRotate(90 , 0, 1, 0)
            
            glCallList(self.axisId)
            glPopMatrix()
            # Y axis, green
            glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE,
                            [0.0, 0.7, 0.0, 1.0])
            glColor4f(0.0, 0.7, 0.0, 1.0)
            glPushMatrix()
            
            glRotate(-90 , 1, 0, 0)
            
            glCallList(self.axisId)
            glPopMatrix()
            # Z axis, blue
            glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE,
                            [0.0, 0.0, 0.7, 1.0])
            #glRotate(-90 , 1, 0, 0)

            glColor4f(0.0, 0.0, 0.7, 1.0)
            glCallList(self.axisId)
            glEndList() 


        def Draw(self, if_draw):
            glPushMatrix()
            if if_draw:
                glTranslate(*self.position)
                '''
                glRotate(-180 * (self.angle[0])/np.pi, 1, 0, 0)
                glRotate(-180 * (self.angle[1])/np.pi, 0,1,0)
                glRotate(-180 * (self.angle[2])/np.pi, 0,0,1)
                '''
                glMultMatrixf(self.Transfer)
                glCallList(self.axisIndicatorId)
            glPopMatrix()



    class DrawPerspectiveLine:
        def __init__(self, position, orientation, up):
            self.fixed_point = np.zeros((1,3))
            self.orientation = orientation
            self.up = up
            self.left = np.cross(self.up, self.orientation)
            self.position = position
            self.length = 600
            self.If_Draw = False
            self.Intersection = np.zeros((3,3)) #[0]与XZ平面交点，[1]与XY平面交点
            self.Distance = np.zeros(3) #[0]与XZ平面dis，[1]与XY平面dis
            self.down_adjust = 0
            
            
        def Find_intersection(self):


            self.up_norm = self.up / np.linalg.norm(self.up)
            self.ori_norm = self.orientation / np.linalg.norm(self.orientation)
            self.left_norm = np.cross(self.up_norm, self.ori_norm)

            self.position = self.position - self.down_adjust * self.up_norm


            #过fixed_point，平行于up的直线与相机所在平面交点
            
            d = -(np.matmul(self.up_norm, self.position.T))
            self.Distance[0] = np.matmul(self.up_norm, self.fixed_point.T) + d
            self.Intersection[0] = self.fixed_point - self.Distance[0] * self.up_norm

            #过fixed_point，平行于ori的直线与视平面交点

            d = -(np.matmul(self.ori_norm, self.position.T))
            self.Distance[2] = np.matmul(self.ori_norm, self.fixed_point.T) + d
            self.Intersection[2] = self.fixed_point - self.Distance[2] * self.ori_norm


            d = -(np.matmul(self.left_norm, self.position.T))
            self.Distance[1] = np.matmul(self.left_norm, self.fixed_point.T) + d
            self.Intersection[1] = self.fixed_point - self.Distance[1] * self.left_norm





        def Draw_Init(self):
            glShadeModel(GL_SMOOTH)
            
            glLineWidth(3.0)
            self.ID = glGenLists(1)
            glNewList(self.ID, GL_COMPILE)
            glBegin(GL_LINES)

            #glColor4f(0.0,1.0,0.0,1.0)
            # glColor4ub(255, 255, 255, 255)
            glColor4ub(255, 97, 0, 255)
            glVertex3f(*(self.Intersection[0]))
            # glColor4ub(255, 255, 255, 255)
            glVertex3f(*self.fixed_point)
            glColor4ub(124, 252, 0, 255)
            glVertex3f(*self.Intersection[1])
            # glColor4ub(255, 255, 255, 255)
            glVertex3f(*self.fixed_point)
            glColor4ub(0, 0, 255, 255)
            glVertex3f(*self.Intersection[2])
            # glColor4ub(255, 255, 255, 255)
            glVertex3f(*self.fixed_point)
            glColor4ub(220, 220, 220, 255)
            glVertex3f(*self.Intersection[0] + self.ori_norm * self.length)
            # glColor4ub(255, 255, 255, 255)
            glVertex3f(*self.Intersection[0] - self.ori_norm * self.length)
            glColor4ub(220, 220, 220, 255)
            glVertex3f(*self.Intersection[0])
            # glColor4ub(255, 255, 255, 255)
            glVertex3f(*self.Intersection[0] - self.left_norm * self.Distance[1])
            glEnd()
            glEndList()
        
        def Draw(self):
            
            glCallList(self.ID)


    class DrawLine:
        def __init__(self, posi0, posi1):
            self.posi0 = posi0
            self.posi1 = posi1
            self.If_Draw = False

        def Draw_Init(self):
            glShadeModel(GL_SMOOTH)

            glLineWidth(2.0)
            self.ID = glGenLists(1)
            glNewList(self.ID, GL_COMPILE)
            glBegin(GL_LINES)

            # posi0 posi1 为顶点的线段
            glColor4ub(255, 0, 0, 255)
            glVertex3f(*self.posi0)
            glColor4ub(255, 255, 255, 255)
            glVertex3f(*self.posi1)
            glEnd()
            glEndList()

        def Draw(self):
            glCallList(self.ID)


    class DrawChar:
        def __init__(self):
            self.CharList = []
            
            self.If_Draw = False




            





    



