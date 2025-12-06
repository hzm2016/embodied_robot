import cv2 as cv
from OpenGL.GLUT import*
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import time 
from numba import *
# from freetype import *
#from threading import Thread


# @jit
# def image_blend(PixelData, newImage, srcImage, rowNumber, colNumber, channels, alpha, beta, gamma):
#     for i in range(rowNumber):
#         for j in range(colNumber):
#             for k in range(channels):
#                 index = ((rowNumber - i - 1) * colNumber + j) * channels + k
#
#                 if (PixelData[index] < 225):
#                     newImage[i, j, k] = srcImage[i, j, k] * alpha + PixelData[index] * beta + gamma
#                     # newImage[i, j, k] = PixelData[index] * 1.0
#
#     return newImage
@jit
def image_blend(PixelData, newImage, srcImage ,rowNumber, colNumber, channels, alpha, beta, gamma):
    for i in range(rowNumber):
        for j in range(colNumber):
            index = ((rowNumber - i - 1) * colNumber + j) * channels
            if (index % 3 == 0): #BGR
                if PixelData[index+2] == 255 and PixelData[index+1] == 97 and PixelData[index] == 0\
                    or PixelData[index+2] == 124 and PixelData[index+1] == 252 and PixelData[index] == 0 \
                    or PixelData[index+2] ==0 and PixelData[index+1] == 0 and 255 >= PixelData[index] >= 255:
                    for k in range(channels):
                        newImage[i, j, k] = PixelData[index+k]

                elif (PixelData[index] < 225 or PixelData[index+1] < 225 or PixelData[index+2] < 225):
                    if (((i > 1 and j > 1) and (i < rowNumber - 1 and j < colNumber - 1))\
                    and ((PixelData[index-3] >=225 and PixelData[index-2] >=225 and PixelData[index-1] >= 225)\
                    or (PixelData[index+3] >=225 and PixelData[index+4] >=225 and PixelData[index+5] >= 225))\
                    or (PixelData[index-colNumber*channels] >=225 and PixelData[index-colNumber*channels] >=225 and PixelData[index-colNumber*channels] >= 225)\
                    or (PixelData[index+colNumber*channels] >=225 and PixelData[index+colNumber*channels] >=225 and PixelData[index+colNumber*channels] >= 225)):
                        newImage[i, j, 0:3] = [24, 74, 46]

                        
                    else:
                        for k in range(channels):
                            newImage[i, j, k] = srcImage[i, j, k] * alpha + PixelData[index+k] * beta + gamma
                
    return newImage



class AR:
    def __init__(self, width, height):
        self.alpha = 0.75
        self.gamma = 0.0
        self.beta = 1 - self.alpha
        self.Width = width
        self.Height = height
        self.source  = "/dev/video_ar"
        self.real = False
        self.success = False
        self.grab_state = False
        self.get_pixel_state = False
        self.Blend_Complete = False
        self.datalist_for_show = ["Null","Null","Null","Null"]
        self.If_ShowData = False
        self.Win_Posi = np.array([[self.Width/2, self.Height/2]],dtype=np.int)
        self.If_ShowLine = False
        self.If_ShowAxis = False
        self.FontLineNum = 6
        self.Dynamic = False

        

    def CV_Init(self, ImageName, real):
        self.FontInit(self.FontLineNum)
        self.real = real
        i = self.Width * 3
        while (i%4):
            i += 1
        self.PixelDataLength = i * self.Height
        self.PixelData = np.zeros(self.PixelDataLength, dtype = np.uint8)
        #self.PixelData = ( GLubyte * self.PixelDataLength )(0)
        
        if real == False:
            self.srcImage = cv.imread(ImageName)
            print(ImageName)
            self.rowNumber = self.srcImage.shape[0]
            self.colNumber = self.srcImage.shape[1]
            self.channels = self.srcImage.shape[2]
            self.targetImage = np.zeros((self.rowNumber, self.colNumber, 3), np.uint8)
            
            self.offset = self.PixelDataLength - self.rowNumber * self.colNumber * self.channels
        else:

            self.cap = cv.VideoCapture(self.source)
            self.cap.set(3, 1280)
            self.cap.set(4, 720)
            self.cap.set(5, 60)
            # self.thread = Thread(target = self.update, args = ())
            # self.thread.daemon = True
            # self.thread.start()

            
            self.success, self.srcImage = self.cap.read()

            if self.success:
                self.srcImage = self.srcImage[0:700, 450:1150, :]
                self.rowNumber = self.srcImage.shape[0]
                self.colNumber = self.srcImage.shape[1]
                self.channels = self.srcImage.shape[2]

                print(self.rowNumber,self.colNumber,self.channels)
                self.targetImage = np.zeros((self.rowNumber, self.colNumber, 3), np.uint8)
                
                self.offset = self.PixelDataLength - self.rowNumber * self.colNumber * self.channels
            else:
                print("Wait for Image")
                #exit(0)

    def update(self):
        while 1:
            if self.cap.isOpened():
                
                #s_t = time.time()
                self.success, self.srcImage = self.cap.read()
                self.srcImage = self.srcImage[0:700,450:1150,:]
                #e_t = time.time()
                #print("\n耗时："+str(e_t - s_t))

    def FontInit(self, num):
        # self.FontWidth = 48
        # self.FontHeight = 64
        # self.face = Face("/home/zwt/Project/first_person_perspective/res/biaoya.TTF")
        # self.face.set_char_size(48 * 64)
        # self.intervalWidth = np.ceil(self.FontWidth/10)
        # self.intervalHeight = np.ceil(self.FontHeight / 10)
        # self.font = [cv.FONT_HERSHEY_SIMPLEX, cv.FONT_HERSHEY_SIMPLEX, cv.FONT_HERSHEY_SIMPLEX, cv.FONT_HERSHEY_SIMPLEX, cv.FONT_HERSHEY_SIMPLEX]
        self.font = [cv.FONT_HERSHEY_SIMPLEX] * num
        # org for the text being specified
        self.org = []
        for i in range(num):
            self.org.append([10, 15 + 25 * i])
        # self.org = [[50, 50], [50, 100], [50, 150], [50, 200], [50, 250], [50, 300]]
        # font scale for the text being specified
        self.fontScale = [0.5] * num
        # Blue color for the text being specified from BGR
        self.color = [[255, 255, 255]] * num
        # Line thickness for the text being specified at 2 px
        self.thickness = [1] * num


    def DatalistInit(self):
        self.datalist_for_show = ["Null","Null","Null","Null"]

    def GenerateOneChar(self, char0):
        self.face.load_char(char0)
        bitmap = self.face.glyph.bitmap
        return bitmap


    def GenerateAuxiliaryData(self, datalist):
        distance_data = datalist[0]
        depth_data = datalist[1]
        dis_data_num = len(distance_data)
        depth_data_num = len(depth_data)
        width_num = max(dis_data_num, depth_data_num)
        height_num = 2
        width = width_num * self.FontWidth + (width_num - 1) * self.intervalWidth
        height = height_num * self.FontHeight + (height_num - 1) * self.intervalHeight
        data_pixel = np.zeros((width, height))
        for i in range(height_num):
            for j in range(width_num):
                bitmap = self.GenerateOneChar(datalist[i][j])




    #@jit
    def Grab_to_CV(self):
        s_t = time.time()
        self.success = False
        if self.real:
            self.success, self.srcImage = self.cap.read()
            self.srcImage = self.srcImage[0:700,450:1150,:]
            if self.success:
                self.newImage = self.srcImage.copy()
            else:
                print("Waiting for Image\n")
                exit(0)
        else:
            self.newImage = self.srcImage.copy()
        #offset = self.offset

        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        #GenerateAuxiliaryData(datalist)
        self.PixelData = glReadPixels(0, 0, self.Width, self.Height, GL_BGR, GL_UNSIGNED_BYTE)
        #self.newImage = image_blend(self.AuxiliaryData, self.PixelData, self.newImage, self.srcImage
        # ,self.rowNumber, self.colNumber, self.channels, self.alpha, self.beta, self.gamma)
        self.newImage = image_blend(self.PixelData, self.newImage, self.srcImage, self.rowNumber,
                                    self.colNumber, self.channels, self.alpha, self.beta, self.gamma)
        if self.If_ShowData:
            self.string_list = ["MidPoint: ","Distance: "+str(self.datalist_for_show[0]),
                                "Coordinate: "+str(self.datalist_for_show[1]),
                                "KeyPoint: ", "Distance: "+str(self.datalist_for_show[2]),
                                "Coordinate: "+str(self.datalist_for_show[3])]
            # self.string_list = ["Distance: "+str(self.datalist_for_show[2]) + "mm"]
            #self.string_list = ["Distance: 35.13mm"]
            for i in range(len(self.string_list)):
                # Using the cv2.putText() method for inserting text in the image of the specified path
                self.newImage = cv.putText(self.newImage, self.string_list[i], self.org[i], self.font[i],
                                           self.fontScale[i], self.color[i], self.thickness[i], cv.LINE_AA)
        if self.If_ShowLine:
            for i in range(len(self.Win_Posi)):
                posi0 = [int(self.Win_Posi[i,0]), int(self.Height - self.Win_Posi[i,1])]
                posi1 = [int(self.Width/2), int(self.Height/2)]
                #print(posi0, posi1)
                cv.line(self.newImage, posi0, posi1, (20,120,50), 2)

        if self.If_ShowAxis:
            # cv.line(self.newImage, [0,int(self.Height/2)], [int(self.Width), int(self.Height/2)], (255, 255, 255), 1)
            cv.line(self.newImage, [0, int(self.Height/2)], [int(self.Width), int(self.Height/2)], (255, 255, 255),1)
            cv.line(self.newImage, [int(self.Width/2),0], [int(self.Width/2), int(self.Height)], (255, 255, 255), 1)

        cv.imwrite(r"G:\first_person_perspective\res\save.png", self.newImage)
        cv.waitKey(0)
        self.newImage = cv.flip(self.newImage, 0, dst=None)
        # if self.Save:
        
        
        





        e_t = time.time()
        #print("\n耗时："+str(e_t - s_t))


    
    def PixelData_to_Matrix(self):
        offset = self.offset
        self.PixelData.reshape(self.rowNumber,-1)
        print(self.PixelData)

            


