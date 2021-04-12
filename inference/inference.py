'''
inference code
'''

import os
import tensorflow as tf
from efficientnet.tfkeras import EfficientNetB0
from PIL import Image
import numpy as np
import cv2

# GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

BATCH_SIZE = 32
IMG_HEIGHT = 224
IMG_WIDTH = 224
AUTOTUNE = tf.data.experimental.AUTOTUNE

infer_root_path = '../data'
files = sorted(os.listdir(infer_root_path))

def load_file_path(files):
    file_ls = []
    for i in files:
        path = os.path.join(infer_root_path,i)
        img = cv2.imread(path)  # reads an image in the BGR format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        data = np.transpose(img, (1, 0, 2))
        data = tf.convert_to_tensor(img, dtype=tf.float32)
        data = tf.cast(data, dtype=tf.uint8)
        data = tf.image.convert_image_dtype(data, tf.float32)
        data = tf.image.resize(data, [IMG_WIDTH, IMG_HEIGHT])
        file_ls.append(data)
    output = tf.data.Dataset.from_tensor_slices(file_ls)
    return output

def img_to_tensor(img):
  return img

infer_list = load_file_path(files)
infer_labeled_ds = infer_list.map(img_to_tensor, num_parallel_calls=AUTOTUNE)

# load model
model_path = '../model.h5'
model = tf.keras.models.load_model(model_path)

result = model.predict(x=infer_labeled_ds.batch(batch_size=BATCH_SIZE), steps=1000, verbose=True)
yhat = np.argmax(result, axis = -1)
# print(yhat)
# print(result)
