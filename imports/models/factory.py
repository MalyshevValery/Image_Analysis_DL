"""Factory of models"""
import copy

from tensorflow.keras.models import Model

from imports.jsonserializable import JSONSerializable
from .unet import UNet


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
