#Importing necessary libraries
import cv2
import numpy as np
import dlib


# import argparse
#
# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--shape-predictor", required=True,
# 	help="path to facial landmark predictor")
# ap.add_argument("-i", "--image", required=True,
# 	help="path to input image")
# args = vars(ap.parse_args())

#Creating a video capture object


cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# predictor = dlib.shape_predictor(args["shape_predictor"])

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        #cv2.rectangle(frame,(x1,y1),(x2,y2), (0,255,0),3)
        landmarks = predictor(gray, face)
        for i in range(0,68):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            cv2.circle(frame,(x,y),3,(255,255,255),-11)
    cv2.imshow('Video', frame)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
