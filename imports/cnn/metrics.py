import tensorflow as tf
from tensorflow.keras import backend as K


def iou(y_true, y_pred, smooth=1.):
    """Intersection over union metric

    l = |intersection| / |union|
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)


def iou_thresholded(y_true, y_pred, threshold=0.5, smooth=1.):
    """Intersection over Union with given threshold applied to y_pred
    """
    ge = tf.greater_equal(y_pred, tf.constant(threshold))
    y_pred = tf.where(ge, x=tf.ones_like(y_pred), y=tf.zeros_like(y_pred))

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)


def dice_coef(y_true, y_pred, smooth=1.):
    """Dice coefficient metric

    l = 2 * |intersecton| / (|A| + |B|)
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
