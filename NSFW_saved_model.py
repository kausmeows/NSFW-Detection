from keras.models import load_model

import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub


class NSFW:
  def __init__(self, img_path, model_path):
    self.img_path = img_path;
    self.model_path = model_path;

  def load_model(self):
    if self.model_path is None or not exists(self.model_path):
    	raise ValueError("saved_model_path must be the valid directory of a saved model to load.")
    
    model = tf.keras.models.load_model(self.model_path)
    return model
  
  def predict_nsfw(self):
    model = load_model(self.model_path)
    img = plt.imread(self.img_path)
    a = smart_resize(img,size=(200,200),interpolation = 'nearest')
    plt.imshow(a)
    a = a.reshape(-1,200,200,3)
    if(np.round(model.predict(a))):
        print('SFW')
    else:
        print('NSFW')


nsfw = NSFW("/content/drive/MyDrive/xxx.jpg", "/content/drive/MyDrive/nsfw_drop.h5")
nsfw.load_model()
nsfw.predict_nsfw()
