# import the necessary packages
import cv2

def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()

# loop over the input images
def detect_blur(image, threshold = 500):
	# load the image, convert it to grayscale, and compute the
	# focus measure of the image using the Variance of Laplacian
	# method
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	fm = variance_of_laplacian(gray)

	# if the focus measure is less than the supplied threshold,
	# then the image should be considered "blurry"
	if fm < threshold:
		return fm
	else:
		return threshold