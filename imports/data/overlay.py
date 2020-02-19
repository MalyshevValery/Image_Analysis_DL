"""Overlays of images by different ground truth"""
import numpy as np


def image_mask(images: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Draw predictions as masks of semantic segmentation on images
    :return: image with drawn blue mask. Currently this is not developed to
        handle mask with more than 3 channels
    """
    if images.shape[-1] == 1:
        images = np.repeat(images, 3, len(images.shape) - 1)
    images[..., -1:-pred.shape[-1] - 1:-1] *= (1.0 - pred)
    return images
