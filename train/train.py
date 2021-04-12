'''
tf.data API로
EfficientNet B0 모델 학습

* Requirement
tensorflow version 2.3.0 이상
'''

import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from efficientnet.tfkeras import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
import datetime
import time
import numpy as np

# GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

BATCH_SIZE = 32
IMG_HEIGHT = 224
IMG_WIDTH = 224
AUTOTUNE = tf.data.experimental.AUTOTUNE


train_path = '../train'
test_path = '../valid'

CLASS_NAMES = np.array(sorted(os.listdir(train_path)))

# 데이터 셋 만들기
train_list_ds = tf.data.Dataset.list_files(str(train_path+'/*/*'),shuffle=True) # 파일 경로를 섞은 데이터셋 반환
test_list_ds = tf.data.Dataset.list_files(str(test_path+'/*/*'),shuffle=False)

# 데이터 셋 확인
for f in train_list_ds.take(5):
  print(f.numpy())

# 데이터 전처리를 위한 데이터 변환
def get_label(file_path):
  parts = tf.strings.split(file_path, os.path.sep)
  # path 의 뒤에서 2번째가 output class 디렉토리
  return parts[-2] == CLASS_NAMES

def decode_img(img):
  # String를 3D uint8 tensor로 변환
  img = tf.image.decode_jpeg(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)
  return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])

def process_path(file_path):
  label = get_label(file_path)
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label

def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
  if cache:
    if isinstance(cache, str):
      ds = ds.cache(cache)
    else:
      ds = ds.cache()

  ds = ds.shuffle(buffer_size=shuffle_buffer_size)
  ds = ds.repeat()
  ds = ds.batch(BATCH_SIZE)
  ds = ds.prefetch(buffer_size=AUTOTUNE)
  return ds

train_labeled_ds = train_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
test_labeled_ds = test_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
train_ds = prepare_for_training(train_labeled_ds)
test_ds = prepare_for_training(test_labeled_ds)

# 1. model
model = Sequential()
model.add(EfficientNetB0(include_top=False, pooling = 'avg', weights = "imagenet"))
model.add(Dense(n, activation='softmax'))
# model.summary()

# checkpoint 저장 path 설정
checkpoint_path = '../weight.ckpt'

modelcheckpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', mode='min', save_best_only=True, save_weights_only=True, verbose=0)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1)

# tensorboard 설정
to_hist = TensorBoard(log_dir='./log' + class_name, histogram_freq=0, write_graph=True, write_images=True)


# 2. compile, fit
model.compile(optimizer = 'adam', loss = tf.keras.losses.categorical_crossentropy, metrics = ['acc'])
hist = model.fit(train_ds,
                 steps_per_epoch=int(len(train_labeled_ds) / BATCH_SIZE),
                 epochs=100,
                 validation_data=test_ds,
                 validation_steps=int(len(test_labeled_ds) / BATCH_SIZE),
                 callbacks=[modelcheckpoint, reduce_lr, time_callback])

model.load_weights(checkpoint_path)

#모델 저장
model_path = '../model.h5'
model.save(model_path)
