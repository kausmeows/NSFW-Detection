from tensorflow.keras.preprocessing.image import ImageDataGenerator, smart_resize
from keras.layers import BatchNormalization
from tensorflow.keras import layers
from tensorflow.keras.utils import image_dataset_from_directory
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, InputLayer
from tensorflow.keras import regularizers
from keras.layers import Dropout
from keras.models import load_model
from tensorflow import keras
from keras.models import Sequential
import numpy as np
import os
import cv2
import keras
import matplotlib.pyplot as plt
import tensorflow as tf
%matplotlib inline

# ---------------------------

class NSFW_CNN:
  def __init__(self, img_path, dataset):
    self.img_path = img_path
    self.dataset = dataset
  
  def img_gen(self):
    i_g = ImageDataGenerator()
    i_g.flow_from_directory(self.dataset)
    train = i_g.flow_from_directory(self.dataset, target_size = (200,200),color_mode='rgb',batch_size = 16, class_mode='binary')
    return train

  def CNNModel():
    model = Sequential()
    model.add(InputLayer(200,200,3))
    model.add(Conv2D(32, (3,3), activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))

    model.add(Conv2D(64, (3,3), activation = 'relu'))
    model.add(MaxPooling2D((2,2)))

    model.add(Conv2D(128, (3,3), activation = 'relu'))
    model.add(MaxPooling2D((2,2)))

    model.add(Flatten())
    model.add(Dense(512, kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.l2(1e-4),
        activity_regularizer=regularizers.l2(1e-5), input_shape = X.shape[1:], activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation = 'relu', kernel_regularizer='l2'))
    model.add(Dense(32, activation = 'relu'))
    model.add(Dense(1,activation='sigmoid'))

    model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
  
    model.fit(train,epochs = 25)

    return model

  def predict_nsfw(self):
    img = plt.imread(self.img_path)
    a = smart_resize(img,size=(200,200),interpolation = 'nearest')
    plt.imshow(a)
    a = a.reshape(-1,200,200,3)
    if(np.round(model.predict(a))):
        return 'SFW'
    else:
        return 'NSFW'

# --------------------------

nsfw = NSFW_CNN('/content/drive/MyDrive/simple.jpg', '/content/drive/MyDrive/nsfw_train')
train = nsfw.img_gen()
model = nsfw.CNNModel()

# Prediction ----------------

nsfw.predict_nsfw()
