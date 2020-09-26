#Importing necessary libraries

import cv2
import numpy as np
import dlib

#Creating face detection and landmark predictor objects using dlib
face_detect = dlib.get_frontal_face_detector()
landmark_predict = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

#Function to convert co-ordinates of landmarks into a numpy array
def to_np_arr(landmarks, dtype="int"):

	coords = np.zeros((68, 2), dtype=dtype)

	for i in range(0, 68):
		coords[i] = (landmarks.part(i).x, landmarks.part(i).y)

	return coords

#Returning a tuple for bounding box
def rect_to_bb(face_rect):
    x = face_rect.left()
    y = face_rect.top()
    w  = face_rect.right() - x
    h = face_rect.bottom() - y
    return (x,y,w,h)

#Creating a video capture object
cap = cv2.VideoCapture(0)

while True:
    #Reading from the webcam
    ret,frame = cap.read()
    if ret == True:
	#Flipping the camera view horizontally to avoid mirror view
        frame = cv2.flip(frame,1)
	#Converting the color image to grayscale for faster computation
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	#Detecting the faces on the grayscale image and getting the co-ordinates
        faces = face_detect(gray)
        for (i,face) in enumerate(faces):
	     #Getting landmark co-ordinates over the detected faces
            landmarks = landmark_predict(gray,face)
	    
            landmarks = to_np_arr(landmarks)
            (x,y,w,h) = rect_to_bb(face)
	     #Drawing rectangle over the detected faces
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(frame,"Face #{}" .format(i+1),(x-10,y-10),cv2.FONT_HERSHEY_PLAIN,0.5,(0,255,0),2)
	     #Drawing circles at all the co-ordinates of the landmarked region
            for (X,Y) in landmarks:
                cv2.circle(frame,(X,Y),2,(255,255,255),-1)
    cv2.imshow('Face-Landmarking',frame)
    k = cv2.waitKey(1) & 0xff
    #Pressing escape to end the program
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
