from PIL import Image
import PIL
import face_recognition
import cv2
import os

def find_faces(image_location, path_of_saved_image = None, save_image = False):
    '''
    :param image_location: Full path to the image
    :param path_of_saved_image: Full path of image with rectangles displaying faces
    :param save_image: True og False, default to False
    :return:
    '''
    # Image path
    image_path = image_location
    new_image_path = os.path.splitext(image_path)[0] + "_reduced.jpg"

    # Create scaled image to increase speed
    fixed_height = 1000
    image = Image.open(image_path)
    height_percent = (fixed_height / float(image.size[1]))
    width_size = int((float(image.size[0]) * float(height_percent)))
    image = image.resize((width_size, fixed_height), PIL.Image.NEAREST)
    image.save(new_image_path)

    # Load the jpg file into a numpy array
    image = face_recognition.load_image_file(new_image_path)
    img_cv = cv2.imread(new_image_path)

    # Find all the faces in the image using a pre-trained convolutional neural network.
    # This method is more accurate than the default HOG model, but it's slower
    # unless you have an nvidia GPU and dlib compiled with CUDA extensions. But if you do,
    # this will use GPU acceleration and perform well.
    # See also: find_faces_in_picture.py
    face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=0, model="cnn")

    # Clean up and delete reduced image
    os.remove(new_image_path)

    # For showing location of faces
    if save_image:
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(img_cv, (left, bottom), (right, top), (255, 0, 0), 2)
        cv2.imwrite(path_of_saved_image, img_cv)

    print("I found {} face(s) in this photograph.".format(len(face_locations)))
    return len(face_locations)
