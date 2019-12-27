"""Factory of models"""
import copy
import os

from tensorflow.keras.models import Model, load_model

from imports.jsonserializable import JSONSerializable
from .conv2dblock import Conv2DBlock
from .unet import UNet

CUSTOMS = {'Conv2DBlock': Conv2DBlock}  # Custom Layers for load_model functions
MODEL_EXT = {'h5', 'tf', 'h5py'}  # Available extensions


class ModelsFactory(JSONSerializable):
    """Factory for creating raw models from JSON configs, saving them and loading models with weights.

    This class can work with other models, not related to this repo till they don't use custom layers. If they do just
    add that layers to CUSTOMS dictionary
    """

    @staticmethod
    def to_json(model: Model):
        """Turns model to JSON"""
        if not hasattr(model, 'meta_info'):
            raise TypeError('Model does not have saved meta info form its creation')
        json = copy.deepcopy(model.meta_info)
        json['name'] = model.name
        return json

    @staticmethod
    def from_json(json, input_shape=None) -> Model:
        """Creates model from JSON"""
        config = copy.deepcopy(json)
        name = config['name']
        del config['name']
        if input_shape is None:
            input_shape = config['input_shape']
            del config['input_shape']
        if name == 'UNet':
            return UNet(input_shape, **config)
        else:
            raise ValueError('Model name ' + name + ' is unknown')

    @staticmethod
    def load(job_dir):
        """Creates model from saved weights and settings (or from saved model)

        :param job_dir: directory with saved data
        """
        files = os.listdir(job_dir)
        if 'best_model.h5' in files:
            return load_model(os.path.join(job_dir, 'best_model.h5'), compile=False, custom_objects=CUSTOMS)
        elif 'model.h5' in files:
            return load_model(os.path.join(job_dir, 'model.h5'), compile=False, custom_objects=CUSTOMS)
        else:
            raise ValueError('No model in job dir')
