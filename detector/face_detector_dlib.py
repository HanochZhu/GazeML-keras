import tensorflow as tf
from keras import backend as K
import numpy as np
import cv2
import os
import dlib

class DlibFaceDetector():
    """
    This class load the dlib network and perform face detection.
    
    Attributes:
        model_path: path to the MTCNN weights files
    """
    def __init__(self,ptsdata_path, model_path = "./models/shape_predictor_68_face_landmarks.dat"):
        self.dlib_path = model_path
        self.ptsdata_path = ptsdata_path
        self.curlandmark_file_path = ""
        
    def create_mtcnn(self, sess, model_path):
        pass
        # detector = dlib.get_frontal_face_detector()
        # predictor = dlib.shape_predictor(self.dlib_path)

    
    def detect_face(self,image_path):
        # print(image_path)
        # _,tmpfilename = os.path.split(image_path)
        # image_name,_ = os.path.splitext(tmpfilename)

        # pts_path = os.path.join(self.ptsdata_path , image_name + ".pts")

        # left [[43,46],[[44,45],[48,47]]]
        # right [[37,40],[[38,39],[41,42]]]
        left_eye_center = []
        right_eye_center = []
        print(self.curlandmark_file_path)
        with open(self.curlandmark_file_path) as pts_io:
            lines = pts_io.readlines()
            coor_43 = DlibFaceDetector.convert_ptsLinesTo_coordinate(lines[115])
            print(coor_43)

            coor_46 = DlibFaceDetector.convert_ptsLinesTo_coordinate(lines[124])
            coor_44 = DlibFaceDetector.convert_ptsLinesTo_coordinate(lines[120])
            coor_45 = DlibFaceDetector.convert_ptsLinesTo_coordinate(lines[121])
            coor_48 = DlibFaceDetector.convert_ptsLinesTo_coordinate(lines[130])
            coor_47 = DlibFaceDetector.convert_ptsLinesTo_coordinate(lines[129])
            coor_t_1 = DlibFaceDetector.get_average_coor(coor_43,coor_46)
            coor_t_2 = DlibFaceDetector.get_average_coor(coor_44,coor_45)
            coor_t_3 = DlibFaceDetector.get_average_coor(coor_47,coor_48)
            coor_t_4 = DlibFaceDetector.get_average_coor(coor_t_2,coor_t_3)
            left_eye_center = DlibFaceDetector.get_average_coor(coor_t_1,coor_t_4)

            
            # right
            coor_37 = DlibFaceDetector.convert_ptsLinesTo_coordinate(lines[135])
            coor_40 = DlibFaceDetector.convert_ptsLinesTo_coordinate(lines[145])
            coor_38 = DlibFaceDetector.convert_ptsLinesTo_coordinate(lines[138])
            coor_39 = DlibFaceDetector.convert_ptsLinesTo_coordinate(lines[139])
            coor_41 = DlibFaceDetector.convert_ptsLinesTo_coordinate(lines[149])
            coor_42 = DlibFaceDetector.convert_ptsLinesTo_coordinate(lines[150])

            coor_t_1 = DlibFaceDetector.get_average_coor(coor_37,coor_40)
            coor_t_2 = DlibFaceDetector.get_average_coor(coor_38,coor_39)
            coor_t_3 = DlibFaceDetector.get_average_coor(coor_41,coor_42)
            coor_t_4 = DlibFaceDetector.get_average_coor(coor_t_2,coor_t_3)
            right_eye_center = DlibFaceDetector.get_average_coor(coor_t_1,coor_t_4)


        return left_eye_center,right_eye_center


    @staticmethod
    def convert_ptsLinesTo_coordinate(line):
        items = line.split(" , ")
        return [int(float(items[0])),int(float(items[1]))]

    @staticmethod
    def get_average_coor(coor_1,coor_2):
        return [(coor_1[0] + coor_2[0]) / 2,(coor_1[1] + coor_2[1]) / 2]

    # @staticmethod
    # def get_eye_average_coor(coor_1,coor_2,coor_3,coor_4,coor_5,coor_6):
    #     DlibFaceDetector.get_average_coor(coor_1)




       

