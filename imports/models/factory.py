"""Factory of models"""
import copy
import os

from tensorflow.keras.models import Model, load_model

from imports.jsonserializable import JSONSerializable
from .conv2dblock import Conv2DBlock
from .unet import UNet

CUSTOMS = {'Conv2DBlock': Conv2DBlock}
MODEL_EXT = {'h5', 'tf', 'h5py'}


class ModelsFactory(JSONSerializable):
    """Model serialization and deserialization from JSON"""

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
    def load(json, job_dir):
        """Creates model from saved weights and settings (or from saved model)

        :param json: settings for creating model
        :param job_dir: directory with saved data
        """
        files = os.listdir(job_dir)
        for ext in MODEL_EXT:
            if 'model.' + ext in files:
                return load_model(os.path.join(job_dir, 'model.' + ext), compile=False, custom_objects=CUSTOMS)

        if 'input_shape' in json:
            model = ModelsFactory.from_json(json)
            if 'weights.h5' in files:
                model.load_weights(os.path.join(job_dir, 'weights.h5'))
            elif 'weights_last.h5' in files:
                model.load_weights(os.path.join(job_dir, 'weights_last.h5'))
            else:
                raise ValueError('No weights in job directory')
            return model
        else:
            raise ValueError('Please provide input shape in settings')
