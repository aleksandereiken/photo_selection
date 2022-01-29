import os

import numpy as np
from PIL import Image
from detect_light import get_light_condition
from import_photos import return_timestamp
from sort_images import group_images
from match_images import match_images_in_groups
from face_detection import find_faces
from classification_model import return_prediction
import tensorflow as tf
# from convert_heic_to_jpg import convert_heic_to_jpeg
from detect_blur import detect_blur
# from other_functions import show_image
import cv2
import shutil
import pandas as pd
pd.set_option("display.max_rows", None, "display.max_columns", None)
import pathlib

# Load prediction model
pred_model = tf.keras.models.load_model("eyes_model")

#if __name__ == '__main__':
subfolder = "other/test_1"
basepath = pathlib.PurePath(os.getcwd(),subfolder).__str__()
images = os.listdir(basepath)

# Convert HEIC images to jpeg, while keeping timestamp intact
# convert_heic_to_jpeg(basepath)

#Create dictonary with paths and timestamps
imgs = []
for image in images:
    with_basepath = basepath + "/" + image
    imgs.append(with_basepath)
good, bad = return_timestamp(imgs)

#Parameters
seconds_between_groups = 30

#Group images based on time between images
image_df = group_images(good, seconds_between_groups)

keep = []
index = 0
while index < (len(image_df)-1):
    print(f"Now working on index {index}")
    first_group = image_df["group"][index]
    second_group = image_df["group"][index + 1]

    # If the two groups above are not identical
    if first_group != second_group:

        #Is the first picture blurry? ### Not working with high res images?
        # blur = detect_blur(cv2.imread(image_df["file_names"][first_group]))
        # if blur > 200:
        print(f"appending image {os.path.basename(image_df['file_names'][index])} to keep")
        keep.append(image_df["file_names"][index])
        #Update index and restart loop
        index = index + 1

    # If the first two groups are identical:
    else:
        #Get DF of all images in same group
        current_group_df = image_df[image_df.group == first_group].reset_index(drop = True)

        # Get number of pictures in group
        n_pictures_in_group = len(current_group_df)

        #Assign subgroups
        cur_gr_df_with_subgr = match_images_in_groups(current_group_df, n_pictures_in_group)

        print(f"For index {index}, there are {len(cur_gr_df_with_subgr)} images with " 
              f" {len(set(cur_gr_df_with_subgr['subgroups']))} group(s)")

        # Select which image to keep within the group
        for row in range(n_pictures_in_group)[:-1]:
            if row == 0:
                comp_img = cur_gr_df_with_subgr["file_names"][0]

            print(f"\tComparator image is now {os.path.basename(comp_img)}")

            #Is row picture and next picture in the same subgroup?
            if cur_gr_df_with_subgr["subgroups"][row] == cur_gr_df_with_subgr["subgroups"][row+1]:
                next_image = cur_gr_df_with_subgr["file_names"][row +1]
                print(f"\tNext image is now {os.path.basename(next_image)}")

                #If row == 0, find faces i comparator image
                if row == 0:
                    n_faces_com, dict_faces_com = find_faces(image_location=comp_img, pers_to_add_to_border=40,
                                                             height_multiplier=2.3)
                    print(f"\tFound {n_faces_com} face(s) in comparator image")

                # Are there any faces in the picture?
                n_faces_1, dict_faces_1 = find_faces(image_location=next_image, pers_to_add_to_border= 40,
                              height_multiplier= 2.3)
                print(f"\tFound {n_faces_1} face(s) in next image")

                diff = n_faces_com - n_faces_1

                # Ensure there are faces in the pictures
                if n_faces_com != 0 or n_faces_1 != 0:

                    # Same amount of faces in each picture?
                    if diff == 0:
                        # Same amount of faces
                        for index_faces in range(n_faces_com):
                            print(f"\t\t Iterating over index_face {index_faces}")
                            if index_faces == 0:
                                cumm_number_com = 0
                                cumm_pct_com = 0
                                cumm_number_1 = 0
                                cumm_pct_1 = 0

                            face_com = Image.fromarray(dict_faces_com.get('face_' + index_faces.__str__()))
                            img_1 = Image.fromarray(dict_faces_1.get('face_' + index_faces.__str__()))

                            number_com, cls_com, pct_com = return_prediction(face_com, pred_model)
                            print(f"\t\t Comparator image with predicted {cls_com} with {pct_com} likelyhood")
                            number_1, cls_1, pct_1 = return_prediction(img_1, pred_model)
                            print(f"\t\t Next image with predicted {cls_1} with {pct_1} likelyhood")

                            cumm_number_com = cumm_number_com + number_com
                            cumm_number_1 = cumm_number_1 + number_1
                            cumm_pct_com = cumm_pct_com + pct_com
                            cumm_pct_1 = cumm_pct_1 + pct_1

                        cumm_diff = cumm_number_com - cumm_number_1
                        cumm_pct = cumm_pct_com - cumm_pct_1

                        #Both are 0, indicating pictures are equally good.
                        if cumm_diff == 0 and abs(cumm_pct) < 40:
                            light_comp_img = get_light_condition(comp_img)
                            light_next_image = get_light_condition(next_image)
                            diff_light = abs(light_comp_img) - abs(light_next_image)
                            if diff_light > 0:
                                n_faces_com = n_faces_1
                                dict_faces_com = dict_faces_1
                                comp_img = next_image
                        elif cumm_diff == 0 and abs(cumm_pct) > 50:
                            if cumm_pct < 0:
                                n_faces_com = n_faces_1
                                dict_faces_com = dict_faces_1
                                comp_img = next_image

                        # Positive ear and positive blur = favour comparator picture
                        elif cumm_diff < 0:
                            n_faces_com = n_faces_1
                            dict_faces_com = dict_faces_1
                            comp_img = next_image

                        # Negative ear and negative blur = favour picture 1
                        elif cumm_diff > 0:
                            continue

                    # n_faces_com has more faces than n_faces_1, i.e. do nothing
                    elif diff > 0:
                        continue

                    # n_faces_1 has more faces than n_faces_com, i.e. update comparator image
                    elif diff < 0:
                        n_faces_com = n_faces_1
                        dict_faces_com = dict_faces_1
                        comp_img = next_image

                else:
                    #Calculate how blury the images are
                    blur_comp_img = detect_blur(cv2.imread(comp_img, cv2.IMREAD_UNCHANGED))
                    blur_next_image = detect_blur(cv2.imread(next_image, cv2.IMREAD_UNCHANGED))

                    #If same amount of blur, calculate light:
                    if abs(blur_comp_img - blur_next_image) < 100:
                        light_comp_img = get_light_condition(comp_img)
                        light_next_image = get_light_condition(next_image)
                        diff = abs(light_comp_img) - abs(light_next_image)
                        if diff > 0:
                            n_faces_com = n_faces_1
                            dict_faces_com = dict_faces_1
                            comp_img = next_image

                    #If next_image is less blury, update comp image
                    elif (blur_comp_img - blur_next_image) < 0:
                        n_faces_com = n_faces_1
                        dict_faces_com = dict_faces_1
                        comp_img = next_image

            else:
                #Add comparator image to keep list
                print(f"Done with a group. Appending image {os.path.basename(comp_img)} to 'keep'")
                keep.append(comp_img)

                #Add new image as comparator image
                comp_img = cur_gr_df_with_subgr["file_names"][row + 1]
                print(f"\tUpdating comparator image to {os.path.basename(comp_img)}")
                n_faces_com, dict_faces_com = find_faces(image_location=comp_img, pers_to_add_to_border=40,
                                                         height_multiplier=2.3)
                print(f"\tFound {n_faces_com} face(s) in comparator image")

        print(f"Appending image {os.path.basename(comp_img)} to 'keep'")
        keep.append(comp_img)
        index = index + n_pictures_in_group

# # #For groups, assign whether images are similar or not
# comparison = CompareImage("images/blurry.jpg", "images/tester.jpg")
# comparison.compare_image()
#
# #For similar images, find faces
# n_faces, dict_faces = find_faces(image_location = "tests/IMG_1332.JPG")
#
# for index in range(n_faces):
#     #Display faces and detect blink
#     img = dict_faces.get('face_' + index.__str__())
#     # Image.fromarray(img).show()
#
    # detect_blur(dict_faces["face_2"])
#     detect_blink(img, EYE_AR_THRESH= 0.20, show_image=True)

# Move images
for image in keep:
    if not os.path.isdir(basepath + "/" + "keep"):
        os.mkdir(basepath + "/" + "keep")
    shutil.move(image, basepath + "/" + "keep/" + image.split(os.sep)[-1])

# Save test results
# textfile = open("results/keep_test_1.txt", "w")
# for element in keep:
#     textfile.write(element + "\n")
# textfile.close()
# img_path_1 = 'subtests/gr_13_1_keep.JPG'
# img_path_2 = 'subtests/gr_3_2.JPG'

### Extract eyes
# import os
# from detect_light import get_light_condition
# from import_photos import return_timestamp
# from sort_images import group_images
# from match_images import match_images_in_groups, image_matching
# from face_detection import find_faces
# from detect_blink import detect_blink
# from detect_blink import save_eyes
# from detect_blur import detect_blur
# from other_functions import show_image
# import cv2
# import pandas as pd
# pd.set_option("display.max_rows", None, "display.max_columns", None)
# import pathlib
# import re
#
# #if __name__ == '__main__':
# subfolder = "pictures"
# basepath = pathlib.PurePath("/Users/aleksandereiken/Documents/backup_fra_stik/spejlrefleks_1",subfolder).__str__()
# images = os.listdir(basepath)
#
# for index, img in enumerate(images):
#     print(f"Working on image index {index} out of {len(images)}")
#     if re.search("\.JPG$|\.jpg$",img):
#         n_faces_com, dict_faces_com = find_faces(image_location=basepath + "/" + img, pers_to_add_to_border=40,
#                                                  height_multiplier=2.3)
#         if n_faces_com:
#             for index, face in enumerate(dict_faces_com):
#                 print(f"face index {index}")
#                 save_eyes(face = dict_faces_com[face],
#                           basepath = basepath + "/eyes",
#                           filename= "eyes_" + index.__str__() + "_" + os.path.basename(img))