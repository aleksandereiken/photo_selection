import tensorflow as tf
import numpy as np
from PIL import Image

def return_prediction(img, pred_model, class_names = ['closed', 'open']):
  #Convert to array
  img_arr = np.asarray(img)

  #Get dimentions and resize
  height, width = img_arr.shape[:2]
  face_cropped_arr = img_arr[int(height * 0.4):int(height * 0.4 + height * 0.25),
                 int(width * 0.1):int(width * 0.1 + width * 0.8)]
  face_cropped = Image.fromarray(face_cropped_arr).resize((180,180))
  img_array = tf.keras.utils.img_to_array(face_cropped)
  img_array = tf.expand_dims(img_array, 0)  # Create a batch
  predictions = pred_model.predict(img_array)
  score = tf.nn.softmax(predictions[0])

  return np.argmax(score), class_names[np.argmax(score)], 100 * np.max(score)