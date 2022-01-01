from PIL import Image
import os
from import_photos import return_timestamp
from sort_images import group_images
from match_images import CompareImage
from face_detection import find_faces
from detect_blink import detect_blink
from convert_heic_to_jpg import convert_heic_to_jpeg

#if __name__ == '__main__':
basepath = "images/"
images = os.listdir(basepath)

# Convert HEIC images to jpeg, while keeping timestamp intact
convert_heic_to_jpeg(basepath)

#Create dictonary with paths and timestamps
imgs = []
for image in images:
    with_basepath = basepath + image
    imgs.append(with_basepath)
good, bad = return_timestamp(imgs)

#Group images based on time between images
image_df = group_images(good)

#For groups, assign whether images are similar or not
comparison = CompareImage("images/blurry.jpg", "images/tester.jpg")
comparison.compare_image()

#For similar images, find faces
n_faces, dict_faces = find_faces(image_location = "images/blurry.jpg")

#Display faces and detect blink
img = dict_faces.get('face_1')
# Image.fromarray(img).show()

detect_blink(img, EYE_AR_THRESH= 0.20, show_image=True)