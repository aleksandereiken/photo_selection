# import the necessary packages
from PIL import Image
from scipy.spatial import distance as dist
from imutils import face_utils
import dlib
import cv2
import numpy
import face_recognition
import os

def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])
	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)
	# return the eye aspect ratio
	return ear

def detect_blink(face, EYE_AR_THRESH = 0.3, show_image = False):

	#Turn into gray
	gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

	### face_recognition library
	landmarks = face_recognition.face_landmarks(gray)
	if landmarks:
		leftEye = numpy.asarray(landmarks[0]["left_eye"])
		rightEye = numpy.asarray(landmarks[0]["right_eye"])

		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)

		# Select average eye value
		ear = (leftEAR + rightEAR) / 2

		if show_image:
			# Draw around the eyes
			leftEyeHull = cv2.convexHull(leftEye)
			rightEyeHull = cv2.convexHull(rightEye)
			cv2.drawContours(face, [leftEyeHull], -1, (0, 255, 0), 1)
			cv2.drawContours(face, [rightEyeHull], -1, (0, 255, 0), 1)
			Image.fromarray(face).show()

		if ear < EYE_AR_THRESH:
			return ear
		else:
			return EYE_AR_THRESH

	else:
		print("No landmarks detected, probably closed eyes")
		return 0

def save_eyes(face, filename, basepath = os.getcwd(), write_image_to_file = True, return_image = False):

		height, width = face.shape[:2]
		face_cropped = face[int(height*0.4):int(height*0.4+height*0.25),int(width * 0.1):int(width * 0.1+width * 0.8)]
		if write_image_to_file:
			cv2.imwrite(f"{basepath}/{filename}", cv2.cvtColor(face_cropped, cv2.COLOR_RGB2BGR))

		if return_image:
			return face_cropped

		# show_image(face_cropped)
		#
		# height_crop_frac = 0.1
		# width_crop_frac = 0.3
		# # Turn into gray
		# gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
		#
		# detector = dlib.get_frontal_face_detector()
		# face_coords = detector(gray, 0)
		#
		# if face_coords:
		# 	### dlib library
		# 	predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
		# 	(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
		# 	(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
		#
		# 	shape = predictor(gray,face_coords[0])
		# 	landmarks = face_utils.shape_to_np(shape)
		#
		# 	if landmarks.size != "0":
		#
		# 		leftEye = landmarks[lStart:lEnd]
		# 		leftEye = leftEye.reshape((-1, 1, 2))
		#
		# 		rightEye = landmarks[rStart:rEnd]
		# 		rightEye = rightEye.reshape((-1, 1, 2))
		#
		# 		max_left = leftEye[3]
		# 		max_right = rightEye[0]
		# 		height = face.shape[0]
		#
		# 		diff = max_left - max_right
		# 		width_add = int(diff[0][0] * width_crop_frac)
		# 		height_add = int(height * height_crop_frac)
		#
		# 		top_left = max_right - [width_add, height_add]
		#
		# 		face_crop = face[top_left[0][1]:top_left[0][1]+height_add*2, top_left[0][0]: top_left[0][0] + diff[0][0] + 2*width_add]
		#
		# 		cv2.imwrite(f"{basepath}/{filename}",cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR))
		# 		# Image.fromarray(face_crop).show()
		#
		# 		# cv2.rectangle(face, top_left[0],buttom_right[0] , (255, 0, 0),1)
		# 		# cv2.polylines(face, [top_left], True, (0, 255, 255))
		#
		# 		# Draw around the eyes
		# 		# leftEyeHull = cv2.convexHull(leftEye)
		# 		# rightEyeHull = cv2.convexHull(rightEye)
		# 		# cv2.drawContours(face, [leftEyeHull], -1, (0, 255, 0), 1)
		# 		# cv2.drawContours(face, [rightEyeHull], -1, (0, 255, 0), 1)
		# 		# Image.fromarray(face_crop).show()
		#
		# 	else:
		# 		print("No eyes found")
		# else:
		# 	print("No landmarks")
