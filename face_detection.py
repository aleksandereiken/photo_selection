import face_recognition
import cv2
import os
from PIL import Image
# image_location = "/Users/aleksandereiken/Documents/python/photo_selection/other/tests/IMG_1346.jpg"

def find_faces(image_location, path_of_saved_image = None, save_faces_to_file = False, save_faces_to_file_basepath = None, save_image_with_faces = False, pers_to_add_to_border = 4, height_multiplier = 1):
    '''
    :param image_location: Full path to the image
    :param path_of_saved_image: Full path of image with rectangles displaying faces
    :param save_image: True og False, default to False
    :return:
    '''
    # Image path
    new_image_path = os.path.splitext(image_location)[0] + "_reduced.jpg"
    # Create scaled image to increase speed
    fixed_height = 2000
    img = cv2.imread(image_location, cv2.IMREAD_COLOR)
    height_percent = (fixed_height / float(img.shape[0]))
    width_size = int((float(img.shape[1]) * float(height_percent)))
    dim = (width_size,fixed_height)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    cv2.imwrite(new_image_path, img)

    # Load the jpg file into a numpy array
    image = face_recognition.load_image_file(new_image_path)
    os.remove(new_image_path)
    # Image.fromarray(img).show()

    face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=0, model="cnn")
    n_faces = len(face_locations)

    if n_faces != 0:
        #Add border to pictures to ensure all face features exits:
        face_locations_mod = []
        for (top, right, bottom, left) in face_locations:

            curr_height = bottom - top
            curr_width = right - left
            increase_width = int(curr_width * (pers_to_add_to_border/100)/2)
            increase_height = int(curr_height * (pers_to_add_to_border/100)/2)

            if top < increase_height * height_multiplier:
                top_mod = 0
            else:
                top_mod = top - int(increase_width*height_multiplier)
            if left < increase_width:
                left_mod = 0
            else:
                left_mod = left - increase_width
            if image.shape[0] > bottom + increase_height:
                bottom_mod = bottom + increase_height
            else:
                bottom_mod = image.shape[0]
            if image.shape[1] > (right + increase_height):
                right_mod = right + increase_width
            else:
                right_mod = image.shape[1]
            face_mod = (top_mod, right_mod,bottom_mod,left_mod)
            face_locations_mod.append(face_mod)

        # For showing location of faces
        faces_dict = dict()
        for (top, right, bottom, left), index in zip(face_locations_mod, range(n_faces)):
            # Get faces and save to dict
            face_image = image[top:bottom, left:right]
            if save_faces_to_file:
                file_name = os.path.splitext(os.path.basename(image_location))[0]
                cv2.imwrite(f"{save_faces_to_file_basepath}/{file_name}_face_{index}.JPG", cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))
            faces_dict[f"face_{index}"] = face_image

        #Save image to see faces recognized??
        if save_image_with_faces:
            # Draw ractangles
            cv2.rectangle(image, (left, bottom), (right, top), (255, 0, 0), 2)
            cv2.imwrite(path_of_saved_image, image)

    if n_faces != 0:
        # print("I found {} face(s) in this photograph.".format(n_faces))
        return(n_faces, faces_dict)
    else:
        # print("No faces found")
        return(0, 0)

def find_faces_from_PIL_img(img, path_of_saved_image = None, save_images = False, show_images = False):
    '''
    :param img: Image of class Image from PIL library
    :param path_of_saved_image: Full path of image with rectangles displaying faces
    :param save_images: True og False, default to False
    :param show_images: True og False, show image in window?
    :return: [n_faces][dict with faces]
    '''
    # Image path
    new_image_path = "pic_reduced.jpg"
    # Create scaled image to increase speed
    fixed_height = 1000
    height_percent = (fixed_height / float(img.shape[0]))
    width_size = int((float(img.shape[1]) * float(height_percent)))
    dim = (width_size,fixed_height)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    cv2.imwrite(new_image_path, img)

    # Load the jpg file into a numpy array
    image = face_recognition.load_image_file(new_image_path)

    if show_images:
        Image.fromarray(img).show()

    face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=0, model="cnn")
    n_faces = len(face_locations)

    # Clean up and delete reduced image
    os.remove(new_image_path)

    # For showing location of faces
    faces_dict = dict()
    for (top, right, bottom, left), index in zip(face_locations, range(n_faces)):
        # Get faces and save to dict
        face_image = image[top:bottom, left:right]
        faces_dict[f"face_{index}"] = face_image

    #Save image to see faces recognized??
    if save_images:
        # Draw ractangles
        cv2.rectangle(image, (left, bottom), (right, top), (255, 0, 0), 2)
        cv2.imwrite(path_of_saved_image, image)

    print("I found {} face(s) in this photograph.".format(n_faces))
    return(n_faces, faces_dict)
