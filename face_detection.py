import face_recognition
import cv2
import os
from PIL import Image

def find_faces(image_location, path_of_saved_image = None, save_images = False):
    '''
    :param image_location: Full path to the image
    :param path_of_saved_image: Full path of image with rectangles displaying faces
    :param save_image: True og False, default to False
    :return:
    '''
    # Image path
    new_image_path = os.path.splitext(image_location)[0] + "_reduced.jpg"
    # Create scaled image to increase speed
    fixed_height = 1000
    img = cv2.imread(image_location, cv2.IMREAD_COLOR)
    height_percent = (fixed_height / float(img.shape[0]))
    width_size = int((float(img.shape[1]) * float(height_percent)))
    dim = (width_size,fixed_height)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    cv2.imwrite(new_image_path, img)

    # Load the jpg file into a numpy array
    image = face_recognition.load_image_file(new_image_path)
    # Image.fromarray(img).show()

    # Find all the faces in the image using a pre-trained convolutional neural network.
    # This method is more accurate than the default HOG model, but it's slower
    # unless you have an nvidia GPU and dlib compiled with CUDA extensions. But if you do,
    # this will use GPU acceleration and perform well.
    # See also: find_faces_in_picture.py
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
