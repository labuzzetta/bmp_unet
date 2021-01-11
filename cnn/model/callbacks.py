import tensorflow as tf
from tensorflow.keras import backend as K
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler, ModelCheckpoint, TensorBoard, Callback

class MyCallback(Callback):
    def __init__(self, alpha):
        self.alpha = alpha
    def on_epoch_end(self, epoch, logs={}):
        self.alpha = np.clip(self.alpha - 0.01, 0.01, 1)

class AlphaScheduler(Callback):
  def __init__(self, alpha, update_fn):
    self.alpha = alpha
    self.update_fn = update_fn
  def on_epoch_end(self, epoch, logs={}):
    updated_alpha = self.update_fn(K.get_value(self.alpha))
