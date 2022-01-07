import os
from PIL import Image
from detect_light import get_light_condition
from import_photos import return_timestamp
from sort_images import group_images
from match_images import match_images_in_groups, image_matching
from face_detection import find_faces
from detect_blink import detect_blink
from convert_heic_to_jpg import convert_heic_to_jpeg
from detect_blur import detect_blur
from other_functions import show_image
import cv2
import shutil
import pandas as pd
pd.set_option("display.max_rows", None, "display.max_columns", None)

#if __name__ == '__main__':
basepath = "subtests/"
images = os.listdir(basepath)

# Convert HEIC images to jpeg, while keeping timestamp intact
convert_heic_to_jpeg(basepath)

#Create dictonary with paths and timestamps
imgs = []
for image in images:
    with_basepath = basepath + image
    imgs.append(with_basepath)
good, bad = return_timestamp(imgs)

#Parameters
seconds_between_groups = 30

#Group images based on time between images
image_df = group_images(good, seconds_between_groups)

keep = []
index = 0
while index < (len(image_df)-1):
    print(index)
    first_group = image_df["group"][index]
    second_group = image_df["group"][index + 1]

    # If the two groups above are not identical
    if first_group != second_group:

        #Is the first picture blurry? ### Not working with high res images?
        # blur = detect_blur(cv2.imread(image_df["file_names"][first_group]))
        # if blur > 200:
        print(f"appending image {image_df['file_names'][index]} to keep")
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

        # Select which image to keep within the group
        for row in range(n_pictures_in_group)[:-1]:
            if row == 0:
                comp_img = cur_gr_df_with_subgr["file_names"][0]

            #Is row picture and next picture in the same subgroup?
            if cur_gr_df_with_subgr["subgroups"][row] == cur_gr_df_with_subgr["subgroups"][row+1]:
                next_image = cur_gr_df_with_subgr["file_names"][row +1]

                # Are there any faces in the picture?
                n_faces_com, dict_faces_com = find_faces(image_location=comp_img, pix_to_add_to_border= 50) #)
                n_faces_1, dict_faces_1 = find_faces(image_location=next_image, pix_to_add_to_border= 50)
                diff = n_faces_com - n_faces_1

                # Ensure there are faces in the pictures
                if n_faces_com != 0 or n_faces_1 != 0:

                    # Same amount of faces in each picture?
                    if diff == 0:
                        # Same amount of faces
                        for index_faces in range(n_faces_com):
                            if index_faces == 0:
                                cumm_ear = 0
                                cumm_blur = 0
                            #Display faces and detect blink
                            face_com = dict_faces_com.get('face_' + index_faces.__str__())
                            img_1 = dict_faces_1.get('face_' + index_faces.__str__())
                            # Image.fromarray(face_com).show()

                            blur_com = detect_blur(face_com)
                            blur_1 = detect_blur(img_1)
                            diff_blur = blur_com - blur_1 # Positive value indicates in favour of picture 0
                            cumm_blur = cumm_blur + diff_blur
                            # print(diff_blur)
                            # print(f"for {index_faces} the cumm_blur is now {cumm_blur}")

                            ear_com = detect_blink(face_com, show_image=False)
                            ear_1 = detect_blink(img_1, show_image=False)
                            diff_ear = ear_com - ear_1
                            cumm_ear = cumm_ear + diff_ear
                            # print(diff_ear)
                            # print(f"for {index_faces} the cumm_ear is now {cumm_ear}")

                        #Both are 0, indicating pictures are equally good.
                        if cumm_ear == 0 and cumm_blur == 0:
                            light_comp_img = get_light_condition(comp_img)
                            light_next_image = get_light_condition(next_image)
                            diff = abs(light_comp_img) - abs(light_next_image)
                            if diff > 0:
                                comp_img = next_image

                        # Positive ear and positive blur = favour comparator picture
                        elif cumm_ear >= 0 and cumm_blur >= 0:
                            continue

                        # Negative ear and negative blur = favour picture 1
                        elif cumm_ear <= 0 and cumm_blur <= 0:
                            comp_img = next_image

                        # Keep comparator image, i.e. do nothing
                        elif cumm_ear <= 0 and cumm_blur >= 0:
                            comp_img = next_image

                        # Positive ear and negative blur = favour picture 1
                        elif cumm_ear >= 0 and cumm_blur <= 0:
                            if(abs(cumm_blur) > 100): #Ensure there is a proper difference in blurriness
                                comp_img = next_image
                            continue

                    # n_faces_com has more faces than n_faces_1, i.e. do nothing
                    elif diff > 0:
                        continue

                    # n_faces_1 has more faces than n_faces_com, i.e. update comparator image
                    elif diff < 0:
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
                            comp_img = next_image

                    #If next_image is less blury, update comp image
                    elif (blur_comp_img - blur_next_image) < 0:
                        comp_img = next_image

            else:
                #Add comparator image to keep list
                print(f"appending image {comp_img} to keep")
                keep.append(comp_img)

                #Add new image as comparator image
                comp_img = cur_gr_df_with_subgr["file_names"][row + 1]

        print(f"appending image {comp_img} to keep")
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

for image in keep:
    if not os.path.isdir(image.split(os.sep)[0] + "/" + "keep/"):
        os.mkdir(image.split(os.sep)[0] + "/" + "keep/")
    shutil.move(image, image.split(os.sep)[0] + "/" + "keep/" + image.split(os.sep)[1])

# Save test results
# textfile = open("results/keep_test_1.txt", "w")
# for element in keep:
#     textfile.write(element + "\n")
# textfile.close()
# img_path_1 = 'subtests/gr_13_1_keep.JPG'
# img_path_2 = 'subtests/gr_3_2.JPG'