"""Model imports"""
import os

from tensorflow.keras.models import load_model

from .conv2dblock import Conv2DBlock
from .unet import UNet

CUSTOMS = {'Conv2DBlock': Conv2DBlock}  # Custom Layers for load_model functions
MODEL_EXT = {'h5', 'tf', 'h5py'}  # Available extensions


def load(model_path):
    """Creates model from saved weights and settings (or from saved model)

    :param model_path: path to model file
    """
    files = os.listdir(model_path)
    if 'best_model.h5' in files:
        return load_model(os.path.join(model_path, 'best_model.h5'),
                          compile=False,
                          custom_objects=CUSTOMS)
    elif 'model.h5' in files:
        return load_model(os.path.join(model_path, 'model.h5'), compile=False,
                          custom_objects=CUSTOMS)
    else:
        raise ValueError('No model in job dir')
