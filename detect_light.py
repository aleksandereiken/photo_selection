import cv2
from PIL import Image

def get_light_condition(image_path, show_image = False):
    fixed_height = 1000
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    height_percent = (fixed_height / float(img.shape[0]))
    width_size = int((float(img.shape[1]) * float(height_percent)))
    dim = (width_size,fixed_height)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.blur(img_gray, (50, 50))  # With kernel size depending upon image size
    if show_image:
        Image.fromarray(blur).show()
    return 127 - cv2.mean(blur)[0]