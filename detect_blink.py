# import the necessary packages
from PIL import Image
from scipy.spatial import distance as dist
from imutils import face_utils
import dlib
import cv2
import numpy

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
	# grab the indexes of the facial landmarks for the left and
	# right eye, respectively
	(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
	(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

	predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

	#Turn into gray
	gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

	#Get coordinates and turn into react
	xcoor_max = numpy.shape(face)[0]
	ycoor_max = numpy.shape(face)[1]
	rect = dlib.rectangle(0, 0, ycoor_max, xcoor_max, )

	# determine the facial landmarks for the face region, then
	# convert the facial landmark (x, y)-coordinates to a NumPy
	# array
	shape = predictor(gray, rect)
	shape = face_utils.shape_to_np(shape)
	# extract the left and right eye coordinates, then use the
	# coordinates to compute the eye aspect ratio for both eyes
	leftEye = shape[lStart:lEnd]
	rightEye = shape[rStart:rEnd]
	leftEAR = eye_aspect_ratio(leftEye)
	rightEAR = eye_aspect_ratio(rightEye)

	# Select eye with maximum value
	ear = max(leftEAR, rightEAR)

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
