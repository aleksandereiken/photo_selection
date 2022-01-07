# import the necessary packages
from PIL import Image
from scipy.spatial import distance as dist
from imutils import face_utils
import dlib
import cv2
import numpy
import face_recognition

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
