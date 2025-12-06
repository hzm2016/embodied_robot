import os
import numpy as np
# import glutils    #Common OpenGL utilities,see glutils.py
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

class ObjModel:
    
    def __init__(self, filepath):
        with open(filepath) as file:
            print("< OBJ >")
            # count = 0
            self.VertexNum = 0
            self.Vertex = []
            self.Normal = []
            self.Face   = []
            self.Index_List  = []
            self.MaxPos = np.zeros((1,3))
            self.MinPos = np.zeros((1,3))
            self.MidPos = np.zeros((1,3))
            j_del = None
            while 1:
                # print(count)
                # count += 1
                line = file.readline()
                if not line:
                    break
                strs = line.split(" ")
                if strs[0] == "v":
                    self.Vertex.append([float(strs[1]), float(strs[2]), float(strs[3])])
                if strs[0] == "vn":
                    self.Normal.append([float(strs[1]), float(strs[2]), float(strs[3])])
                if strs[0] == "f":
                    for i in range(1, len(strs)):  # 将四边形切分为三角面片
                        element = strs[i].strip("\n").split("/") 
                        # print(element)
                        if i == 1:
                            for j in range(len(element)):
                                if element[j] == '':
                                    j_del = j
                            if j_del is not None:
                                del element[j_del]
                        else:
                            # print(j_del)
                            if j_del is not None:
                                del element[j_del]  
                        self.Index_List.append(element[0:len(strs)])
                        for i, index_unit in enumerate(self.Index_List):
                            for j, index in enumerate(index_unit):
                                self.Index_List[i][j] = int(index)
                    #  print(self.Index)
                        
            self.VertexNum = len(self.Index_List)        
            self.Vertex = np.array(self.Vertex)
            self.Normal = np.array(self.Normal)
            for i in range(len(self.Vertex)):
                if i == 0:
                    for j in range(3):
                        self.MaxPos[0,j] = self.MinPos[0,j] = self.Vertex[0,j]
                else:
                    if self.MaxPos[0].sum() < self.Vertex[i].sum():
                        self.MaxPos[0] = self.Vertex[i]
                    if self.MinPos[0].sum() > self.Vertex[i].sum():
                        self.MinPos[0] = self.Vertex[i]
        self.MidPos = (self.MaxPos + self.MinPos) / 2.0
        print(self.MinPos)
        print(self.MaxPos)
        print(self.MidPos)

    def draw_Init(self):
        # self.program = glutils.loadShaders(strVS, strFS)
        # glUseProgram(self.program)
        # set up vertex array object (VAO)
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        # set up VBOs
        vertexData = np.array(self.Vertex, np.float32)
        self.vertexBuffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vertexBuffer)
        glBufferData(GL_ARRAY_BUFFER, 4*len(vertexData), vertexData, 
                     GL_STATIC_DRAW)
        #enable arrays
        self.vertIndex = 0
        glEnableVertexAttribArray(self.vertIndex)
        # set buffers 
        glBindBuffer(GL_ARRAY_BUFFER, self.vertexBuffer)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        # unbind VAO
        glBindVertexArray(0)

    def render(self):
        # # use shader
        # glUseProgram(self.program)
        # bind VAO
        glBindVertexArray(self.vao)
        # draw
        glDrawArrays(GL_TRIANGLES, 0, 3)
        # unbind VAO
        glBindVertexArray(0)


    def Draw_Init(self):

        #self.index = GLuint()
        #self.offset = np.array((59.41809383,-28.68455701,-95.25279284))

        self.index = glGenLists(1)
        Dict = {}

        for i, index_unit in enumerate(self.Index_List):
            Vertex_index = index_unit[0] - 1 #一个顶点的空间坐标索引
            Normal_index = index_unit[1] #一个顶点法向量的索引
            Normal_vector = self.Normal[Normal_index - 1] #一个顶点的法向量
            if Vertex_index in Dict:
                Dict[Vertex_index] += -Normal_vector
            else:
                Dict[Vertex_index] = -Normal_vector


        
        glEnable(GL_NORMALIZE)
        #print(type(self.Vertex))
        glNewList(self.index, GL_COMPILE)
        glBegin(GL_TRIANGLES)
        for i, index_unit in enumerate(self.Index_List):
            Vertex_index = index_unit[0] - 1 #一个顶点的空间坐标索引
            glNormal3f(*Dict[Vertex_index])
            # print(Dict[Vertex_index])
            glVertex3f(*self.Vertex[Vertex_index])
            # print(self.Vertex[Vertex_index])
        glEnd()
        glEndList()
        


    def Draw(self,if_draw, main_body):

        glPushMatrix()
        # glTranslatef(*-self.MidPos[0])
        if if_draw:
            #glPushMatrix()
            glCallList(self.index)
        glPopMatrix()
