from import_photos import return_timestamp
from sort_images import group_images
from match_images import CompareImage
from face_detection import find_faces
import os

#if __name__ == '__main__':
basepath = "images/"
images = os.listdir(basepath)

#Create dictonary with paths and timestamps
imgs = []
for image in images:
    with_basepath = basepath + image
    imgs.append(with_basepath)
good, bad = return_timestamp(imgs)

#Group images based on time between images
image_df = group_images(good)

#For groups, assign whether images are similar or not
comparison = CompareImage("images/os_alle_1.jpg", "images/os_alle.jpg")
comparison.compare_image()

#For similar images, find faces
find_faces(image_location= "images/esben_1 copy.jpg")
