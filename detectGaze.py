# CPU inference
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from detector.face_detector import MTCNNFaceDetector
from models.elg_keras import KerasELG
from keras import backend as K

import numpy as np
import cv2
from matplotlib import pyplot as plt

def draw_pupil(lms):
    pupil_center = np.zeros((2,))
    pnts_outerline = []
    pnts_innerline = []
    stroke = 1
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
def recalPos(pos,imagew,imageh):
    _recalPos = []
    for i,_pos in enumerate(pos):
        _recalPos.append([int(_pos[0] / 180 * imagew),int(_pos[1] / 108 * imageh)])
    return _recalPos


def detect_iris(image_path):
        

    mtcnn_weights_dir = "./mtcnn_weights/"
    fd = MTCNNFaceDetector(sess=K.get_session(), model_path = mtcnn_weights_dir)

    model = KerasELG()
    model.net.load_weights("./elg_weights/elg_keras.h5")

    input_img = cv2.imread(image_path)[..., ::-1]

    face, lms = fd.detect_face(input_img) # assuming there is only one face in input image
    assert len(face) >= 1, "No face detected"

    left_eye_xy = np.array([lms[6], lms[1]])
    right_eye_xy = np.array([lms[5], lms[0]])

    dist_eyes = np.linalg.norm(left_eye_xy - right_eye_xy)
    # bounding box
    eye_bbox_w = (dist_eyes / 1.25)
    eye_bbox_h = (eye_bbox_w * 0.6)

    # draw = input_img.copy()

    ###
    #  ——————————————
    # |             |
    # |             |
    # |      *      |
    # |             |
    # |_____________|
    ###

    left_eye_im = input_img[
        int(left_eye_xy[0]-eye_bbox_h//2):int(left_eye_xy[0]+eye_bbox_h//2),
        int(left_eye_xy[1]-eye_bbox_w//2):int(left_eye_xy[1]+eye_bbox_w//2), :]


    #left_eye_im = left_eye_im[:,::-1,:] # No need for flipping left eye for iris detection
    right_eye_im = input_img[
        int(right_eye_xy[0]-eye_bbox_h//2):int(right_eye_xy[0]+eye_bbox_h//2),
        int(right_eye_xy[1]-eye_bbox_w//2):int(right_eye_xy[1]+eye_bbox_w//2), :]

    inp_left = cv2.cvtColor(left_eye_im, cv2.COLOR_RGB2GRAY)
    inp_left = cv2.equalizeHist(inp_left)
    #  (180,108) ： 宽 and 高
    inp_left = cv2.resize(inp_left, (180,108))[np.newaxis, ..., np.newaxis]

    inp_right = cv2.cvtColor(right_eye_im, cv2.COLOR_RGB2GRAY)
    inp_right = cv2.equalizeHist(inp_right)
    inp_right = cv2.resize(inp_right, (180,108))[np.newaxis, ..., np.newaxis]


    input_array = np.concatenate([inp_left, inp_right], axis=0)

    pred_left, pred_right = model.net.predict(input_array/255 * 2 - 1)

    lms_left = model._calculate_landmarks(pred_left)
    lms_right = model._calculate_landmarks(pred_right)

    # 拼接
    input_array = np.concatenate([inp_left, inp_right], axis=0)
    pred_left, pred_right = model.net.predict(input_array/255 * 2 - 1)

    # g and b iris and pupil
    # hm_g = np.max(pred_left[...,8:16], axis=-1, keepdims=True)
    # hm_b = np.max(pred_left[...,16:], axis=-1, keepdims=True)
    # hm_g = np.max(pred_right[...,8:16], axis=-1, keepdims=True)
    # hm_b = np.max(pred_right[...,16:], axis=-1, keepdims=True)

    lms_left = model._calculate_landmarks(pred_left)

    lms_right = model._calculate_landmarks(pred_right)

    lms_left = model._calculate_landmarks(pred_left)
    actual_pos_left = draw_pupil(lms_left)

    # x,y
    actual_pos_left = recalPos(actual_pos_left,eye_bbox_w,eye_bbox_h)

    lms_right = model._calculate_landmarks(pred_right)
    actual_pos_right = draw_pupil(lms_right)

    actual_pos_right = recalPos(actual_pos_right,eye_bbox_w,eye_bbox_h)

    # for lm in iris_lm_right:
    #     draw = cv2.circle(draw, (int(left_bboxx + lm[0]),int(left_bboxy + lm[1])), 1, (255,255,255), -1)

    left_eye_im_leftTop = [int(left_eye_xy[0]-eye_bbox_h//2),int(left_eye_xy[1]-eye_bbox_w//2)]

    # left_eye_im_leftBottom = [int(left_eye_xy[0] + eye_bbox_h//2),int(left_eye_xy[1]-eye_bbox_w//2)]
    # left_eye_im_rightTop = [int(left_eye_xy[0]-eye_bbox_h//2),int(left_eye_xy[1]+ eye_bbox_w//2)]
    # left_eye_im_rightBottom = [int(left_eye_xy[0]+eye_bbox_h//2),int(left_eye_xy[1]+eye_bbox_w//2)]

    right_eye_im_leftTop = [int(right_eye_xy[0]-eye_bbox_h//2),int(right_eye_xy[1]-eye_bbox_w//2)]

    # for i, lm in enumerate([left_eye_xy, right_eye_xy]):
    #     draw = cv2.circle(draw, (int(lm[1]), int(lm[0])), 1, (255*i,255*(1-i),0), -1)

    # draw = cv2.circle(draw, (100,0), 10, (255,0,0), -1)
    #draw = cv2.circle(draw, (left_eye_im_leftTop[1], left_eye_im_leftTop[0]), 1, (255,0,0), -1)
    #draw = cv2.circle(draw, (left_eye_im_leftBottom[1], left_eye_im_leftBottom[0]), 1, (0,255,0), -1)
    # draw = cv2.circle(draw, (left_eye_im_rightTop[1], left_eye_im_rightTop[0]), 1, (0,0,255), -1)
    #draw = cv2.circle(draw, (left_eye_im_rightBottom[1], left_eye_im_rightBottom[0]), 1, (255,255,255), -1)

    iris_lm_right = []
    iris_lm_left = []

    for lm in actual_pos_left[8:16]:
        iris_lm_left.append([int(left_eye_im_leftTop[1] + lm[0] ),int(left_eye_im_leftTop[0] + lm[1])])

    for lm in actual_pos_right[8:16]:
        iris_lm_right.append([int(right_eye_im_leftTop[1] + lm[0]),int(right_eye_im_leftTop[0] + lm[1])])

    # only iris landmark
    return iris_lm_left,iris_lm_right

def test():
    
    fn = "./test_imgs/Lenna_(test_image).png"

    actual_pos_left,actual_pos_right = detect_iris(fn)
    draw = cv2.imread(fn)[..., ::-1]

    for lm in actual_pos_left:
        draw = cv2.circle(draw, (lm[0],lm[1]), 1, (255,255,255), -1)

    for lm in actual_pos_right:
        draw = cv2.circle(draw, ( lm[0], lm[1]), 1, (255,255,255), -1)

    cv2.imshow("image",draw)

    cv2.waitKey()
        
test()