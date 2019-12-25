# CPU inference
import os

print("import")
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from detector.face_detector import MTCNNFaceDetector
from models.elg_keras import KerasELG
from keras import backend as K

import numpy as np
import cv2
from matplotlib import pyplot as plt


mtcnn_weights_dir = "./mtcnn_weights/"
fd = MTCNNFaceDetector(sess=K.get_session(), model_path=mtcnn_weights_dir)

model = KerasELG()
model.net.load_weights("./elg_weights/elg_keras.h5")

fn = "./test_imgs/Lenna_(test_image).png"
input_img = cv2.imread(fn)[..., ::-1]

face, lms = fd.detect_face(input_img) # assuming there is only one face in input image
assert len(face) >= 1, "No face detected"

left_eye_xy = np.array([lms[6], lms[1]])
right_eye_xy = np.array([lms[5], lms[0]])

dist_eyes = np.linalg.norm(left_eye_xy - right_eye_xy)
# bounding box
eye_bbox_w = (dist_eyes / 1.25)
eye_bbox_h = (eye_bbox_w * 0.6)

draw = input_img.copy()

left_bboxx = left_eye_xy[0]-eye_bbox_h//2
left_bboxy = left_eye_xy[1]-eye_bbox_h//2

###
#  ——————————————
# |             |
# |      .      |
# |             |
# |             |
# _______________
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

inp_right = cv2.cvtColor(right_eye_im, cv2.COLOR_RGB2GRAY)
inp_right = cv2.equalizeHist(inp_right)

input_array = np.concatenate([inp_left, inp_right], axis=0)

print(input_array/255 * 2 - 1)
pred_left, pred_right = model.net.predict(input_array/255 * 2 - 1)

lms_left = model._calculate_landmarks(pred_left)
lms_right = model._calculate_landmarks(pred_right)

input_array = np.concatenate([inp_left, inp_right], axis=0)
pred_left, pred_right = model.net.predict(input_array/255 * 2 - 1)

# g and b iris and pupil
hm_g = np.max(pred_left[...,8:16], axis=-1, keepdims=True)
hm_b = np.max(pred_left[...,16:], axis=-1, keepdims=True)

hm_g = np.max(pred_right[...,8:16], axis=-1, keepdims=True)
hm_b = np.max(pred_right[...,16:], axis=-1, keepdims=True)

lms_left = model._calculate_landmarks(pred_left)

for i in lms_left:
    print(i)

lms_right = model._calculate_landmarks(pred_right)


for i, lm in enumerate([left_eye_xy, right_eye_xy]):
    draw = cv2.circle(draw, (int(lm[1]), int(lm[0])), 1, (255*i,255*(1-i),0), -1)
    print(int(lm[1]), int(lm[0])) # x,y

for lm in lms_left[0]:
    draw = cv2.circle(draw, (int(left_bboxx + lm[0] * 3 ),int(left_bboxy + lm[1] * 3)), 1, (255,255,255), -1)

cv2.imshow("image",draw)

cv2.waitKey()
    