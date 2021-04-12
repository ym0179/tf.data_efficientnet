'''
test code
'''

import os
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from efficientnet.tfkeras import EfficientNetB0

# GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# params
BATCH_SIZE = 32
IMG_HEIGHT = 224
IMG_WIDTH = 224
AUTOTUNE = tf.data.experimental.AUTOTUNE


def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[-2] == CLASS_NAMES


def decode_img(img):
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])


def process_path(file_path):
    label = get_label(file_path)
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label

def process_path2(file_path):
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img

def tensor_to_array(array_value):
    return array_value.numpy()


# load model
model_path = '../model.h5'
model = tf.keras.models.load_model(model_path)

# test dataset path
infer_path = '../test'
infer_list_ds = tf.data.Dataset.list_files(str(infer_path + '/*'), shuffle=False)
infer_labeled_ds = infer_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

# 정답 라벨 추출
arr = np.array([])
for img, label in iter(infer_labeled_ds):
    arr = np.append(arr, int(np.argmax(tensor_to_array(label))))

infer_labeled_ds = infer_list_ds.map(process_path2, num_parallel_calls=AUTOTUNE) # without label

# predict
result = model.predict(x=infer_labeled_ds.batch(batch_size=BATCH_SIZE), steps=1000, verbose=True)
yhat = np.argmax(result, axis=-1)

# calculate accuracy
acc = accuracy_score(yhat, arr)
acc_ls.append(acc)
