"""Directory storage"""
import copy
import os

import numpy as np
from cv2 import cv2

from .abstract import AbstractStorage

COLOR_TRANSFORMS = {
    'none': lambda image: image,
    'to_gray': lambda image: cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis],
    'from_gray': lambda image: cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else
    cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
}


class DirectoryStorage(AbstractStorage):
    """Directory storage that loads all PNG images in directory"""

    def __init__(self, data_dir, color_transform='none', mode='r'):
        """Constructor

        :param data_dir: directory with PNG images
        """
        self._dir = data_dir
        if mode is 'w':
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            super().__init__(mode='w')
        elif mode is 'r':
            if not os.path.isdir(data_dir):
                raise ValueError(data_dir + ' is not a directory')

            all_names = os.listdir(data_dir)
            png_names = [name for name in all_names if name.endswith('.png')]
            png_filenames = [name for name in png_names if os.path.isfile(os.path.join(data_dir, name))]

            super().__init__(keys=set(png_filenames), mode=mode)

        if color_transform not in COLOR_TRANSFORMS:
            raise ValueError(color_transform + ' is not a valid color representation')
        self._color_transform = COLOR_TRANSFORMS[color_transform]
        self._color_transform_label = color_transform

    def __getitem__(self, item):
        super().__getitem__(item)
        image = cv2.imread(os.path.join(self._dir, item))
        return self._color_transform(image)

    def save_array(self, keys, array):
        """Saves array to directory specified in init"""
        super().save_array(keys, array)
        for i in range(len(keys)):
            cv2.imwrite(os.path.join(self._dir, keys[i]), self._color_transform(array[i]))

    @classmethod
    def type(cls):
        """Returns type of this decorator"""
        return 'directory'

    def to_json(self):
        """Returns python dict to transform to JSON"""
        return {
            'type': DirectoryStorage.type(),
            'dir': self._dir,
            'color_transform': self._color_transform_label
        }

    @staticmethod
    def from_json(json, mode='r'):
        """Returns object generator from dict"""
        config = copy.deepcopy(json)
        if json['type'] != DirectoryStorage.type():
            raise ValueError('Type ' + json['type'] + ' is invalid type for DirectoryStorage')
        del config['dir']
        del config['type']
        return DirectoryStorage(json['dir'], **config)