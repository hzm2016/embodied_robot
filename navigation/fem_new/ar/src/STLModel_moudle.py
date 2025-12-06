from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
#import glm as glm
import numpy as np
import struct
#import glfw as glfw

 
class STLModel:

    def __init__(self, filename, form):
        if form:
            print("\n<<< STL deb >>>\n")
            try:
                file = open(filename, 'rb')
            except:
                print("reading file failed")
                exit(0)
            file_header = file.read(80)
            self.Triangle_num = struct.unpack('i', file.read(sizeof(int32_t)))[0]
            self.Vertex = np.zeros((3 * self.Triangle_num, 3))
            self.Normal = np.zeros((3 * self.Triangle_num, 3))
            self.Property = np.zeros((self.Triangle_num, 1))
            self.MaxPos = np.zeros((1,3))
            self.MinPos = np.zeros((1,3))
            self.MidPos = np.zeros((1,3))
            #####记录一个三角形面片的3个法向量坐标
            for i in range(self.Triangle_num):
                temp = struct.unpack(12 *'f',file.read(12 * sizeof(int32_t)))
                self.Normal[3*i,:] = temp[0:3] 
                #####记录一个三角形面片的三个顶点坐标
                for j in range(3):
                    ###记录1个顶点的三维坐标
                    self.Vertex[3*i+j,:] = temp[3*(j+1):3*(j+1)+3]
                    #虽然获取的是int字节长度的数据，但实际数据代表的是float
                ###记录一个三角形面片的属性信息
                file.read(2)

                if i == 0:
                    for j in range(3):
                        self.MaxPos[0,j] = self.MinPos[0,j] = self.Vertex[0,j]
                for j in range(3):
                    for k in range(3):
                        if self.MaxPos[0,j] < self.Vertex[3*i+k,j]:
                            self.MaxPos[0,j] = self.Vertex[3*i+k,j]
                        if self.MinPos[0,j] > self.Vertex[3*i+k,j]:
                            self.MinPos[0,j] = self.Vertex[3*i+k,j]
            file.close()

            for i in range(self.Triangle_num):
                self.Normal[3*i+1] = self.Normal[3*i]
                self.Normal[3*i+2] = self.Normal[3*i]
            self.MidPos = (self.MaxPos + self.MinPos) / 2.0
        else:
            print("\n<<< STL ASCII >>>\n")
    
    
    def Eliminate_Duplicates(self):
        print("\n<<< Eliminate Duplicates >>>\n")


    def Draw_Init(self):

        #self.index = GLuint()
        #self.offset = np.array((59.41809383,-28.68455701,-95.25279284))

        self.index = glGenLists(1)
        Dict = {}
        for i in range(len(self.Vertex)):
            
            if tuple(self.Vertex[i,:]) in Dict:
                Dict[tuple(self.Vertex[i,:])] += -self.Normal[i,:].copy()
                #Dict[tuple(self.Vertex[i,:])] = np.vstack([Dict[tuple(self.Vertex[i,:])],self.Normal[i,:]])
            else:
                Dict[tuple(self.Vertex[i,:])] = -self.Normal[i,:].copy()
        
        glEnable(GL_NORMALIZE)
        #print(type(self.Vertex))
        glNewList(self.index, GL_COMPILE)
        glBegin(GL_TRIANGLES)
        for i in range(self.Triangle_num):
            for j in range(3):
                glNormal3f(*Dict[tuple(self.Vertex[3*i+j,:])])
                #glNormal3f(*-self.Normal[3*i+j,:])
                glVertex3f(*self.Vertex[3*i+j,:])
        glEnd()
        glEndList()
        


    def Draw(self,if_draw, main_body):

        glPushMatrix()

        # if main_body == 0:
        #     self.offset = np.array((57-119.5154705,-75-138.55568695,-35-105.82824516))
        #     glTranslatef(*self.offset)
        # else:
        #     glTranslatef(-self.MidPos[0,0], -self.MidPos[0,1], -self.MidPos[0,2])
        # if main_body == 0:
        #     glTranslatef(*self.offset)
        if if_draw:
            #glPushMatrix()
            glCallList(self.index)
        glPopMatrix()


    def Draw1(self,if_draw, main_body):
        glEnable(GL_NORMALIZE)
        glPushMatrix()
        if main_body == 1:
            self.MidPos = (self.MaxPos + self.MinPos) / 2.0
        else:
            glTranslatef(57,-75,-35)
        glTranslatef(-self.MidPos[0,0], -self.MidPos[0,1], -self.MidPos[0,2])
        glBegin(GL_POINTS)
        if if_draw:
            for i in range(2 * self.Triangle_num):
                for j in range(3):
                    glNormal3f(*-self.Normal[3*i+j,:])
                    glVertex3f(*self.Vertex[3*i+j,:])
                    if i != 2 * self.Triangle_num - 1:
                        glNormal3f(*(-self.Normal[3*i+j,:] - self.Normal[3*i+j+1,:]))
                        glVertex3f(*(self.Vertex[3*i+j,:] + self.Vertex[3*i+j+1,:]))
        glEnd()
        glPopMatrix()


            
