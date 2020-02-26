"""Model imports"""
import os
from typing import Tuple

from tensorflow.keras.models import load_model, Model

from .conv2dblock import Conv2DBlock
from .test_net import TestNet
from .unet import UNet

CUSTOMS = {'Conv2DBlock': Conv2DBlock}  # Custom Layers for load_model functions


def load(job_dir: str,
         try_names: Tuple[str, ...] = ('best_model.h5', 'model.h5')) -> Model:
    """Creates model from saved weights and settings (or from saved model)

    :param job_dir: Path to model file
    :param try_names: Names to try to load
    :raises FileNotFoundError if no file from try_names was found
    """
    files = os.listdir(job_dir)
    for f in try_names:
        if f in files:
            return load_model(os.path.join(job_dir, f), CUSTOMS, False)
    raise FileNotFoundError('No model in job dir')
