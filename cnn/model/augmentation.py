import tensorflow as tf
from tensorflow.keras import backend as K
import cv2
import numpy as np
import math
import os
import params
import re
import tifffile
import random
import glob

def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.uniform(hue_shift_limit[0], hue_shift_limit[1])
        h = cv2.add(h, hue_shift)
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image

def randomHorizontalFlip_2(image1, image2, mask, u=0.5):
    if np.random.random() < u:
        image1 = cv2.flip(image1, 1)
        image2 = cv2.flip(image2, 1)
        mask = cv2.flip(mask, 1)

    return image1, image2, mask

def randomVerticalFlip_2(image1, image2, mask, u=0.5):
    if np.random.random() < u:
        image1 = cv2.flip(image1, 0)
        image2 = cv2.flip(image2, 0)
        mask = cv2.flip(mask, 0)

    return image1, image2, mask

def randomHorizontalFlip_1(image1, mask, u=0.5):
    if np.random.random() < u:
        image1 = cv2.flip(image1, 1)
        mask = cv2.flip(mask, 1)

    return image1, mask

def randomVerticalFlip_1(image1, mask, u=0.5):
    if np.random.random() < u:
        image1 = cv2.flip(image1, 0)
        mask = cv2.flip(mask, 0)

    return image1, mask
