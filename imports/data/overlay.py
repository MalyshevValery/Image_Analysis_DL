"""Overlays of images by different ground truth"""
import numpy as np


def image_mask(images, pred):
    """Draw predictions as masks on images
    :return: image with drawn blue mask
    """
    if images.shape[-1] == 1:
        images = np.repeat(images, 3, len(images.shape) - 1)
    images[:, :, :, -1:-pred.shape[-1] - 1:-1] *= (1.0 - pred)
    return images
