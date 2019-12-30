# CPU inference
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import sys
sys.path.append(r"D:\Work\Git\MoreFaceLandmark\Utils\GazeMLkeras")
sys.path.append(r".\detector")

# from detector.face_detector import MTCNNFaceDetector
from detector.face_detector_dlib import DlibFaceDetector
from models.elg_keras import KerasELG
from keras import backend as K

import numpy as np
import cv2
from matplotlib import pyplot as plt


class DetectGaze:

    def __init__(self,h5_path,pts_path = "../../faces/data"):

        self.curlandmark_file_path = ""
        self.detector = DlibFaceDetector(pts_path)
        self.model = KerasELG()
        self.model.net.load_weights(h5_path)

    def draw_pupil(self,lms):
        actual_pos = []
        # 0-7 eye corner
        # 8-15 iris
        # 16 pupil
        for i, lm in enumerate(np.squeeze(lms)):
            #print(lm)
            # y 就是x，x就是y
            y, x = int(lm[0] * 3), int(lm[1]*3)
            actual_pos.append([y,x])
        return actual_pos

    # xy
    def recalPos(self,pos,imagew,imageh):
        _recalPos = []
        for i,_pos in enumerate(pos):
            _recalPos.append([int(_pos[0] / 180 * imagew),int(_pos[1] / 108 * imageh)])
        return _recalPos


    def detect_iris(self,image_path):
        
        input_img = cv2.imread(image_path)[..., ::-1]
        self.detector.curlandmark_file_path = self.curlandmark_file_path
        left_eye_center,right_eye_center = self.detector.detect_face(image_path)
        # print(left_eye_center)
        # print(right_eye_center)
        left_eye_center = [int(left_eye_center[0]),int(left_eye_center[1])]
        right_eye_center = [int(right_eye_center[0]),int(right_eye_center[1])]
        # left_eye_xy = np.array([lms[6], lms[1]])
        # right_eye_xy = np.array([lms[5], lms[0]])

        left_eye_xy = np.array([left_eye_center[0], left_eye_center[1]])
        right_eye_xy = np.array([right_eye_center[0], right_eye_center[1]])
    


        dist_eyes = np.linalg.norm(left_eye_xy - right_eye_xy)
        # bounding box
        eye_bbox_w = (dist_eyes / 1.25)
        eye_bbox_h = (eye_bbox_w * 0.6)
        # print(dist_eyes)
        # print(eye_bbox_w)
        # print(eye_bbox_h)

        # draw = input_img.copy()

        ###
        #  ——————————————
        # |             |
        # |             |
        # |      *      |
        # |             |
        # |_____________|
        ###

        # left_eye_xy.astype(int)
        # right_eye_xy.astype(int)
        left_eye_im = input_img[
            int(left_eye_xy[1] - eye_bbox_h//2):int(left_eye_xy[1] + eye_bbox_h//2),
            int(left_eye_xy[0]- eye_bbox_w//2):int(left_eye_xy[0] + eye_bbox_w//2), :]

        #left_eye_im = left_eye_im[:,::-1,:] # No need for flipping left eye for iris detection
        right_eye_im = input_img[
            int(right_eye_xy[1] - eye_bbox_h//2):int(right_eye_xy[1] + eye_bbox_h//2),
            int(right_eye_xy[0] - eye_bbox_w//2):int(right_eye_xy[0] + eye_bbox_w//2), :]

        # draw = input_img.copy()
        # draw = cv2.circle(draw, (left_eye_center[0], left_eye_center[1]), 5, (0,0,255), -1)
        # draw = cv2.circle(draw, (right_eye_center[0], right_eye_center[1]), 5, (0,255,0), -1)
        # draw = cv2.circle(draw, (int(left_eye_xy[0] - eye_bbox_w//2), int(left_eye_xy[1]- eye_bbox_h//2)), 5, (255,255,255), -1)
        # draw = cv2.circle(draw, (int(left_eye_xy[0] + eye_bbox_w//2), int(left_eye_xy[1]- eye_bbox_h//2)), 5, (255,255,255), -1)
        # draw = cv2.circle(draw, (int(left_eye_xy[0] - eye_bbox_w//2), int(left_eye_xy[1]+ eye_bbox_h//2)), 5, (255,255,255), -1)
        # draw = cv2.circle(draw, (int(left_eye_xy[0] + eye_bbox_w//2), int(left_eye_xy[1]+ eye_bbox_h//2)), 5, (255,255,255), -1)

        # draw = cv2.circle(draw, (int(right_eye_xy[0] - eye_bbox_w//2), int(right_eye_xy[1]- eye_bbox_h//2)), 5, (255,255,255), -1)
        # draw = cv2.circle(draw, (int(right_eye_xy[0] + eye_bbox_w//2), int(right_eye_xy[1]- eye_bbox_h//2)), 5, (255,255,255), -1)
        # draw = cv2.circle(draw, (int(right_eye_xy[0] - eye_bbox_w//2), int(right_eye_xy[1]+ eye_bbox_h//2)), 5, (255,255,255), -1)
        # draw = cv2.circle(draw, (int(right_eye_xy[0] + eye_bbox_w//2), int(right_eye_xy[1]+ eye_bbox_h//2)), 5, (255,255,255), -1)
        # draw = cv2.resize(draw,(draw.shape[1] // 2,draw.shape[0] // 2))

        # cv2.imshow("image",draw)

        # cv2.waitKey()
        # draw = right_eye_im.copy()
        
        # cv2.imshow("image",draw)

        # cv2.waitKey()
        # draw = left_eye_im.copy()
        
        # cv2.imshow("image",draw)

        # cv2.waitKey()


        inp_left = cv2.cvtColor(left_eye_im, cv2.COLOR_RGB2GRAY)
        inp_left = cv2.equalizeHist(inp_left)
        #  (180,108) ： 宽 and 高
        inp_left = cv2.resize(inp_left, (180,108))[np.newaxis, ..., np.newaxis]

        inp_right = cv2.cvtColor(right_eye_im, cv2.COLOR_RGB2GRAY)
        inp_right = cv2.equalizeHist(inp_right)
        inp_right = cv2.resize(inp_right, (180,108))[np.newaxis, ..., np.newaxis]


        input_array = np.concatenate([inp_left, inp_right], axis=0)

        pred_left, pred_right = self.model.net.predict(input_array/255 * 2 - 1)

        lms_left = self.model._calculate_landmarks(pred_left)
        lms_right = self.model._calculate_landmarks(pred_right)

        # 拼接
        input_array = np.concatenate([inp_left, inp_right], axis=0)
        pred_left, pred_right = self.model.net.predict(input_array/255 * 2 - 1)

        # g and b iris and pupil

        lms_left = self.model._calculate_landmarks(pred_left)

        lms_right = self.model._calculate_landmarks(pred_right)

        lms_left = self.model._calculate_landmarks(pred_left)
        actual_pos_left = self.draw_pupil(lms_left)

        # x,y
        actual_pos_left = self.recalPos(actual_pos_left,eye_bbox_w,eye_bbox_h)

        lms_right = self.model._calculate_landmarks(pred_right)
        actual_pos_right = self.draw_pupil(lms_right)

        actual_pos_right = self.recalPos(actual_pos_right,eye_bbox_w,eye_bbox_h)

        left_eye_im_leftTop = [int(left_eye_xy[0] - eye_bbox_w//2),int(left_eye_xy[1] - eye_bbox_h//2)]

        right_eye_im_leftTop = [int(right_eye_xy[0] - eye_bbox_w//2),int(right_eye_xy[1] - eye_bbox_h//2)]


        iris_lm_right = []
        iris_lm_left = []
        # iris landmark index [8:16]
        for lm in actual_pos_left:
            
            iris_lm_left.append([int(left_eye_im_leftTop[0] + lm[0] ),int(left_eye_im_leftTop[1] + lm[1])])
            # print(lm)

        for lm in actual_pos_right:
            iris_lm_right.append([int(right_eye_im_leftTop[0] + lm[0]),int(right_eye_im_leftTop[1] + lm[1])])

        
        # draw = input_img.copy()
        # draw = cv2.circle(draw, (left_eye_center[0], left_eye_center[1]), 5, (0,0,255), -1)
        # draw = cv2.circle(draw, (right_eye_center[0], right_eye_center[1]), 5, (0,255,0), -1)

        # draw = cv2.circle(draw, (left_eye_im_leftTop[0],left_eye_im_leftTop[1]), 5, (255,255,0), -1)
        # draw = cv2.circle(draw, (right_eye_im_leftTop[0],right_eye_im_leftTop[1]), 5, (255,255,0), -1)

        # draw = cv2.circle(draw, (iris_lm_left[0][0],iris_lm_left[0][1]), 5, (0,0,255), -1)
        # # draw = cv2.circle(draw, (iris_lm_right[0][0],iris_lm_right[0][1]), 5, (0,0,255), -1)
        # draw = cv2.resize(draw,(draw.shape[1] // 3,draw.shape[0] // 3))

        # cv2.imshow("image",draw)

        # cv2.waitKey()

        # only iris landmark
        return iris_lm_left,iris_lm_right


def test():
    
    # fn = "./test_imgs/Lenna_(test_image).png"
    # fn = "./2449179046_1.jpg"
    fn = "./2397891505_1.jpg"
    
    mtcnn_weights_dir = "./mtcnn_weights/"
    h5_path = "./elg_weights/elg_keras.h5"
    pts_path = "../../faces/data"

    D = DetectGaze(h5_path,pts_path)
    D.curlandmark_file_path = "./2397891505_1.txt"
    actual_pos_left,actual_pos_right = D.detect_iris(fn)
    draw = cv2.imread(fn)[..., ::-1]
    # draw = cv2.resize(draw,(1024,1024))

    for i,lm in enumerate(actual_pos_left[8:17]):
        draw = cv2.circle(draw, (lm[0] ,lm[1] ), 1, (255,255,255), -1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        draw = cv2.putText(draw, str(i+1),(lm[0],lm[1]),font, 0.3, (0, 0, 255), 1, cv2.LINE_AA)

    for i,lm in enumerate(actual_pos_right):
        draw = cv2.circle(draw, ( lm[0], lm[1]), 1, (255,255,255), -1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        draw = cv2.putText(draw, str(i+1),(lm[0] ,lm[1]),font, 0.3, (0, 0, 255), 1, cv2.LINE_AA)
    
    # draw = cv2.resize(draw,(draw.shape[1] // 3,draw.shape[0] // 3))
    cv2.imshow("image",draw)

    cv2.waitKey()

# test()