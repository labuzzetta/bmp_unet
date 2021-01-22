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

#Add extra convolution to convert 4 channel lidar model to 3 channels for imagenet compatibility
base_model = Unet(backbone_name='resnet50', encoder_weights='imagenet', decoder_use_batchnorm=True)
inp = Input(shape=(None, None, 4))
l1 = Conv2D(3, (1,1))(inp)
out = base_model(l1)
model = Model(inp, out, name=base_model.name)

ids_total = [s.split("_")[8] for s in glob.glob("/work/LAS/zhuz-lab/clabuzze/grassed_waterway_test_500/hillshade_clipped_IA_halfFishs/grassed_waterway/*.tif")]

metric_f = FScore()
metric_i = IOUScore()
# returns a compiled model
# identical to the previous one
model = load_model('/work/LAS/zhuz-lab/clabuzze/model_wrr_gw_lidar_imagenet.h5', custom_objects={'image_dice_coeff':image_dice_coeff, 'image_dice_coeff_round':image_dice_coeff_round, 'global_dice_coef':global_dice_coef, 'global_dice_loss':global_dice_loss, 'cldice':cldice, 'combined_loss':combined_loss, 'matthews_correlation':matthews_correlation, 'recall_m':recall_m, 'precision_m':precision_m, 'f1-score':metric_f, 'iou_score':metric_i})

for id in ids_total:
    
    x_pred_batch = []

    img = cv2.imread('/work/LAS/zhuz-lab/clabuzze/grassed_waterway_test_500/hillshade_clipped_IA_halfFishs/grassed_waterway/fishGrid_{}_lidar_hs_1m.tif'.format(id))
    img = cv2.resize(img, (input_size, input_size))
    img0 = img
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.expand_dims(img, axis=2)

    img2 = cv2.imread('/work/LAS/zhuz-lab/clabuzze/grassed_waterway_test_500/ortho_clipped_IA_halfFishs_2007_2010/grassed_waterway/fishGrid_{}_ortho_2007_2010_cir.tif'.format(id))
    img2 = cv2.resize(img2, (input_size, input_size))

    mask = cv2.imread('/work/LAS/zhuz-lab/clabuzze/grassed_waterway_test_500/gw_raster_clipped_IA_halfFishs/fishGrid_{}.tif'.format(id))
    mask0 = cv2.resize(mask, (input_size, input_size))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    mask = cv2.resize(mask, (input_size, input_size))
    mask = np.expand_dims(mask, axis=2)

    mask[mask <= 128] = 0
    mask[mask > 128] = 1

    img_stack = np.concatenate((img, img2), axis=2)
    x_pred_batch.append(img_stack)
    x_pred_batch = np.array(x_pred_batch, np.float32) / 255
    
    pred_out = model.predict(x_pred_batch)

    im_pred = np.array(255*pred_out[0],dtype=np.uint8)

    rgb_mask_pred = cv2.cvtColor(im_pred,cv2.COLOR_GRAY2RGB)
    rgb_mask_pred = cv2.cvtColor(rgb_mask_pred,cv2.COLOR_RGB2GRAY)
    (thresh, img_pred) = cv2.threshold(rgb_mask_pred, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    img_pred[img_pred>0] = 1

    y_batch = []
    y_batch.append(mask)
    y_pred_batch = np.array(y_batch, np.float32)

    results = model.evaluate(x_pred_batch, y_pred_batch, verbose=1)
    print(str(id) + str(",") + str(results))

    img_pred = np.stack((255*img_pred, 255*img_pred, 255*img_pred), axis=-1)
    vis = img_pred

    vis1 = np.concatenate((mask0,img0),axis=0)
    vis2 = np.concatenate((img2,img_pred),axis=0)
    vis = np.concatenate((vis1,vis2),axis=1)
    
    #cv2.imwrite('/work/LAS/zhuz-lab/clabuzze/preprocess_classification_BMP_centroid/truth/te_truth_{}.png'.format(id), vis)    
    #cv2.imwrite('/work/LAS/zhuz-lab/clabuzze/preprocess_classification_BMP_centroid/PRO_valid/gw_truth_{}_{}.png'.format(id, round(dice_val,2)), vis)
