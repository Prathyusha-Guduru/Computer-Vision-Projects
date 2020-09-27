
import cv2
import numpy as np
from collections import OrderedDict
import dlib



face_landmark = OrderedDict([
	("mouth", (48, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 36)),
	("jaw", (0, 17))
])


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


def get_facial_regions(frame, landmarks, colors=None, alpha=0.75):
	global face_landmark
	overlay = frame.copy()
	output = frame.copy()

	if colors is None:
		colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23),
				  (168, 100, 168), (158, 163, 32),
				  (163, 38, 32), (180, 42, 220)]
	for (i, name) in enumerate(face_landmark.keys()):
		(j,k) = face_landmark[name]
		pts = landmarks[j:k]

		if name == 'jaw' :
			for l in range(1,len(pts)):
				pt_prev = tuple(pts[l-1])
				pt_next = tuple(pts[l])
				cv2.line(overlay,pt_prev,pt_next,colors[i],2)
		else:
			hull = cv2.convexHull(pts)
			cv2.drawContours(overlay,[hull],-1,colors[i],-1)
	cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
	return output

image = cv2.imread('Hans-solo.jpg')
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
image_copy = image.copy()



face_detect = dlib.get_frontal_face_detector()
landmark_predict = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
# image = cv2.cvtColor(image, cv2.COLOR_)

faces = face_detect(gray)
for (i, face) in enumerate(faces):

	landmarks = landmark_predict(gray, face)
	landmarks = to_np_arr(landmarks)


	for (name, (i,j)) in face_landmark.items():
		copy = image.copy()
		cv2.putText(copy, name,(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

		for(x,y) in landmarks[i:j]:
			cv2.circle(copy, (x, y), 1, (0, 0, 255), -1)
			(x,y,w,h) = cv2.boundingRect(np.array([landmarks[i:j]]))
			roi = image[y:y+h, x:x+w]
			cv2.namedWindow('ROI', cv2.WINDOW_NORMAL)
			cv2.resizeWindow('ROI', 250,100)
			cv2.imshow('ROI', roi)


			cv2.imshow('Image',copy)
			cv2.waitKey(0)

		Result = get_facial_regions(image, landmarks)
		cv2.imshow('Facial Regions', Result)
		cv2.waitKey(0)


