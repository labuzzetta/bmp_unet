from __future__ import print_function, division
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.utils import to_categorical
import numpy as np

smooth = 1
epsilon = 1e-5

def auc(y_true, y_pred):
    auc = tf.keras.metrics.AUC(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

#Batch-wise mean of recall
#Values are rounded to 0 or 1 to find true positives
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

#Batch-wise mean of precision
#Values are rounded to 0 or 1 to find true positives and predicted positives
def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

#Implementation of Image Based dice loss, calcaulated over each image
#and then the average of the dice coefficient for each image is taken
def image_dice_coeff(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)

#Values are rounded to 0 or 1 to formally calculate matrix of true positives, false positives, ...
def image_dice_coeff_round(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * tf.cast(K.greater(y_pred, 0.5),tf.float32), axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(tf.cast(K.greater(y_pred, 0.5),tf.float32), axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)

def image_dice_loss(y_true, y_pred):
    return 1-image_dice_coeff(y_true, y_pred)

#Implementation of Global Dice Loss, calculated over all images in batch
#Helps to avoid discontinuities of image-based dice loss
def global_dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def global_dice_loss(y_true, y_pred):
    loss = 1 - global_dice_coef(y_true, y_pred)
    return loss

def matthews_correlation(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())

def cl_dice_loss(data_format="channels_first"):
    """dice loss function for tensorflow/keras
        calculate dice loss per batch and channel of each sample.
    Args:
        data_format: either channels_first or channels_last
    Returns:
        loss_function(y_true, y_pred)  
    """

    def loss(target, pred):
        if data_format == "channels_last":
            pred = tf.transpose(pred, (0, 3, 1, 2))
            target = tf.transpose(target, (0, 3, 1, 2))

        smooth = 1.0
        iflat = tf.reshape(
            pred, (tf.shape(pred)[0], tf.shape(pred)[1], -1)
        )  # batch, channel, -1
        tflat = tf.reshape(target, (tf.shape(target)[0], tf.shape(target)[1], -1))
        intersection = K.sum(iflat * tflat, axis=-1)
        return 1 - ((2.0 * intersection + smooth)) / (
            K.sum(iflat, axis=-1) + K.sum(tflat, axis=-1) + smooth
        )

    return loss

def soft_skeletonize(x, thresh_width=10):
    """
    Differenciable aproximation of morphological skelitonization operaton
    thresh_width - needs to be greater then or equal to the maximum radius for the tube-like structure
    """

    minpool = (
        lambda y: K.pool2d(
            y * -1,
            pool_size=(3, 3),
            strides=(1, 1),
            pool_mode="max",
            data_format="channels_first",
            padding="same",
        )
        * -1
    )
    maxpool = lambda y: K.pool2d(
        y,
        pool_size=(3, 3),
        strides=(1, 1),
        pool_mode="max",
        data_format="channels_first",
        padding="same",
    )

    for i in range(thresh_width):
        min_pool_x = minpool(x)
        contour = K.relu(maxpool(min_pool_x) - min_pool_x)
        x = K.relu(x - contour)
    return x

def norm_intersection(center_line, vessel):
    """
    inputs shape  (batch, channel, height, width)
    intersection formalized by first ares
    x - suppose to be centerline of vessel (pred or gt) and y - is vessel (pred or gt)
    """
    smooth = 1.0
    clf = tf.reshape(
        center_line, (tf.shape(center_line)[0], tf.shape(center_line)[1], -1)
    )
    vf = tf.reshape(vessel, (tf.shape(vessel)[0], tf.shape(vessel)[1], -1))
    intersection = K.sum(clf * vf, axis=-1)
    return (intersection + smooth) / (K.sum(clf, axis=-1) + smooth)

def soft_cldice_loss(k=10, data_format="channels_first"):
    """clDice loss function for tensorflow/keras
    Args:
        k: needs to be greater or equal to the maximum radius of the tube structure.
        data_format: either channels_first or channels_last        
    Returns:
        loss_function(y_true, y_pred)  
    """

    def loss(target, pred):
        if data_format == "channels_last":
            pred = tf.transpose(pred, (0, 3, 1, 2))
            target = tf.transpose(target, (0, 3, 1, 2))

        cl_pred = soft_skeletonize(pred, thresh_width=k)
        target_skeleton = soft_skeletonize(target, thresh_width=k)
        iflat = norm_intersection(cl_pred, target)
        tflat = norm_intersection(target_skeleton, pred)
        intersection = iflat * tflat
        return 1 - ((2.0 * intersection) / (iflat + tflat))

    return loss

def cldice_loss(y_true, y_pred):
    return soft_cldice_loss(k=10, data_format="channels_last")(y_true, y_pred)

def cldice(y_true, y_pred):
    return 1 - soft_cldice_loss(k=10, data_format="channels_last")(y_true, y_pred)

def combined_loss(y_true, y_pred):
    alpha = 0.5
    return (alpha * cl_dice_loss(data_format="channels_last")(y_true, y_pred) + (1-alpha) * soft_cldice_loss(k=10, data_format="channels_last")(y_true, y_pred))
