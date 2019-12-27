"""Callbacks wrapper to JSON serializable"""
import copy
import json
import os
from typing import List

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, Callback

from imports.jsonserializable import JSONSerializable

# Available callbacks
CALLBACK_MAP = {
    'early_stop': EarlyStopping,
    'tensorboard': TensorBoard,
    'checkpoint': ModelCheckpoint
}


class CallbacksWrapper(JSONSerializable):
    """Wrapper for tensorflow callbacks JSON serialization, deserialization.
    Consult CALLBACK_MAP to see available callbacks.
    """

    def __init__(self, json_params, base_dir=None):
        """Constructor

        :param json_params: parameters of desired callbacks written in JSON
        :param base_dir: directory in which all stuff from callbacks (weights, logs) will be saved
        """
        config = copy.deepcopy(json_params)
        self._config_json = copy.deepcopy(json_params)
        if not isinstance(config, list):
            config = [config]

        str_config = json.dumps(config)
        if base_dir is None and '$BASE$' in str_config:
            raise ValueError('base dir is None but $BASE$ is in config')
        str_config = str_config.replace('$BASE$', base_dir)
        config = json.loads(str_config)

        self._callbacks = []
        for c in config:
            if not isinstance(c, dict):
                raise TypeError(str(c) + " is not a dictionary with params")
            params = copy.deepcopy(c)
            name = params.get('name', None)
            del params['name']

            if name != 'checkpoint':
                self._callbacks.append(CALLBACK_MAP[name](**params))
            else:
                self._callbacks.append(CALLBACK_MAP[name](os.path.join(base_dir, 'weights.h5'), **params))

    def get_callbacks(self) -> List[Callback]:
        """Returns created callbacks"""
        return self._callbacks

    def to_json(self):
        """JSON representation of callback"""
        return self._config_json

    @staticmethod
    def from_json(json_config, base_dir=None):
        """Get callbacks from JSON"""
        return CallbacksWrapper(json_config, base_dir)
