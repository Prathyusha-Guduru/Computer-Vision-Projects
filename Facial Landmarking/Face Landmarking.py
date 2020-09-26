#Importing necessary libraries

import cv2
import numpy as np
import dlib

face_detect = dlib.get_frontal_face_detector()
landmark_predict = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def to_np_arr(landmarks, dtype="int"):

	coords = np.zeros((68, 2), dtype=dtype)

	for i in range(0, 68):
		coords[i] = (landmarks.part(i).x, landmarks.part(i).y)

	return coords

def rect_to_bb(face_rect):
    x = face_rect.left()
    y = face_rect.top()
    w  = face_rect.right() - x
    h = face_rect.bottom() - y
    return (x,y,w,h)

cap = cv2.VideoCapture(0)

while True:
    ret,frame = cap.read()
    if ret == True:
        frame = cv2.flip(frame,1)
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_detect(gray)
        for (i,face) in enumerate(faces):
            landmarks = landmark_predict(gray,face)
            landmarks = to_np_arr(landmarks)
            (x,y,w,h) = rect_to_bb(face)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(frame,"Face #{}" .format(i+1),(x-10,y-10),cv2.FONT_HERSHEY_PLAIN,0.5,(0,255,0),2)

            for (X,Y) in landmarks:
                cv2.circle(frame,(X,Y),2,(255,255,255),-1)
    cv2.imshow('Face-Landmarking',frame)
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break
