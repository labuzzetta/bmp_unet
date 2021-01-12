import tensorflow as tf
import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler, ModelCheckpoint, TensorBoard, Callback, LambdaCallback
from model.callbacks import MyCallback, AlphaScheduler
from model.losses import image_dice_coeff, image_dice_coeff_round, global_dice_coef, global_dice_loss, cldice, combined_loss, matthews_correlation, recall_m, precision_m
from model.augmentation import randomHueSaturationValue, randomHorizontalFlip_2, randomVerticalFlip_2, randomHorizontalFlip_1, randomVerticalFlip_1
from tensorflow.keras.optimizers import Adam
import math
import os
import params
import re
import tifffile
import random
import glob

input_size = 1024

epochs = 100

batch_size = 5

from segmentation_models import Unet, Linknet
from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.models import Model
from segmentation_models.metrics import FScore, IOUScore

#Encoder weights to None for baseline model
model = Unet(backbone_name='resnet50', encoder_weights=None, decoder_use_batchnorm=True, input_shape=(None, None, 3))

ids_train_split = [s.split("_")[8] for s in glob.glob("/work/LAS/zhuz-lab/clabuzze/pond_dam_sys_1000/hillshade_clipped_IA_halfFishs/pond_dam/*.tif")]

ids_valid_split = [s.split("_")[8] for s in glob.glob("/work/LAS/zhuz-lab/clabuzze/pond_dam_sys_500/hillshade_clipped_IA_halfFishs/pond_dam/*.tif")]

def train_generator():
    while True:
 
        for start in range(0, len(ids_train_split), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(ids_train_split))
            ids_train_batch = ids_train_split[start:end]
            for id in ids_train_batch:
                
                img = cv2.imread('/work/LAS/zhuz-lab/clabuzze/pond_dam_sys_1000/hillshade_clipped_IA_halfFishs/pond_dam/fishGrid_{}_lidar_hs_1m.tif'.format(id))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                try:
                    img = cv2.resize(img, (input_size, input_size))
                except Exception as e:
                    print(str(e))
                
                img2 = cv2.imread('/work/LAS/zhuz-lab/clabuzze/pond_dam_sys_1000/ortho_clipped_IA_halfFishs_2007_2010/pond_dam/fishGrid_{}_ortho_2007_2010_cir.tif'.format(id))
                try:
                    img2 = cv2.resize(img2, (input_size, input_size)) 
                except Exception as e:
                    print(str(e))

                mask = cv2.imread('/work/LAS/zhuz-lab/clabuzze/pond_dam_sys_1000/pd_raster_clipped_IA_halfFishs/fishGrid_{}.tif'.format(id))
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                
                try:
                    mask = cv2.resize(mask, (input_size, input_size))
                except Exception as e:
                    print(str(e))
                
                mask[mask <= 128] = 0
                mask[mask > 128] = 1 

                img2 = randomHueSaturationValue(img2,
                                               hue_shift_limit=(-30, 30),
                                               sat_shift_limit=(-5, 5),
                                               val_shift_limit=(-15, 30))
            
                img, img2, mask = randomHorizontalFlip_2(img, img2, mask)
                
                img, img2, mask = randomVerticalFlip_2(img, img2, mask)

                img = np.expand_dims(img, axis=2)
                mask = np.expand_dims(mask, axis=2)
                
                img_stack = np.concatenate((img, img2), axis=2)
                #DO NOT INCLUDE LIDAR IN THIS MODEL, only img2
                x_batch.append(img2)
                y_batch.append(mask)

            x_batch = np.array(x_batch, np.float32) / 255
            y_batch = np.array(y_batch, np.float32)    # no need to divide 255, alread 0-1
            yield x_batch, y_batch

def valid_generator():
    while True:
        for start in range(0, len(ids_valid_split), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(ids_valid_split))
            ids_valid_batch = ids_valid_split[start:end]
            for id in ids_valid_batch:

                img = cv2.imread('/work/LAS/zhuz-lab/clabuzze/pond_dam_sys_500/hillshade_clipped_IA_halfFishs/pond_dam/fishGrid_{}_lidar_hs_1m.tif'.format(id))
                try:
                    img = cv2.resize(img, (input_size, input_size))
                except Exception as e:
                    print(str(e))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = np.expand_dims(img, axis=2)
                
                img2 = cv2.imread('/work/LAS/zhuz-lab/clabuzze/pond_dam_sys_500/ortho_clipped_IA_halfFishs_2007_2010/pond_dam/fishGrid_{}_ortho_2007_2010_cir.tif'.format(id))
                try:
                    img2 = cv2.resize(img2, (input_size, input_size))   
                except Exception as e:
                    print(str(e))  

                mask = cv2.imread('/work/LAS/zhuz-lab/clabuzze/pond_dam_sys_500/pd_raster_clipped_IA_halfFishs/fishGrid_{}.tif'.format(id))
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                
                try:
                    mask = cv2.resize(mask, (input_size, input_size))
                    mask = np.expand_dims(mask, axis=2)
                except Exception as e:
                    print(str(e))

                mask[mask <= 128] = 0
                mask[mask > 128] = 1

                img_stack = np.concatenate((img, img2), axis=2)
                #DO NO INCLUDE LIDAR IN THIS MODEL, only img2
                x_batch.append(img2)
                y_batch.append(mask)

            x_batch = np.array(x_batch, np.float32) / 255
            y_batch = np.array(y_batch, np.float32)  # no need to divide 255, alread 0-1
            yield x_batch, y_batch

metric_f = FScore()
metric_i = IOUScore()
model.compile(optimizer=Adam(0.001), loss=global_dice_loss, metrics=["accuracy", image_dice_coeff, image_dice_coeff_round, global_dice_coef, cldice, matthews_correlation, tf.keras.metrics.AUC(), recall_m, precision_m, metric_f, metric_i])

callbacks = [ReduceLROnPlateau(monitor='val_loss',
                               factor=0.1, 
                               patience=10,
                               verbose=1,
                               min_lr = 1e-7),
             TensorBoard(log_dir='logs')]

model.fit_generator(generator=train_generator(),
                    steps_per_epoch=np.ceil(float(len(ids_train_split)) / float(batch_size)),
                    epochs=epochs,
                    verbose=2,
                    callbacks=callbacks,
                    validation_data=valid_generator(),
                    validation_steps=np.ceil(float(len(ids_valid_split)) / float(batch_size)))

model.save('/work/LAS/zhuz-lab/clabuzze/model_wrr_pd_nolidar.h5')
