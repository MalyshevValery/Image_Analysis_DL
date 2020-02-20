"""Directory storage"""
import os
from typing import Set

import numpy as np
from cv2 import cv2

from .abstract import AbstractStorage, Mode


class DirectoryStorage(AbstractStorage):
    """Directory storage that loads all **PNG** images in directory

    :param directory: directory with PNG images
    :param gray_transform: True to transform images to grayscale after reading
        and before writing
    :param mode: Mode.READ for read Mode.WRITE for write
    """

    def __init__(self, directory: str,
                 gray_transform: bool = False,
                 mode: Mode = Mode.READ):
        self.__dir = directory
        if mode is Mode.WRITE:
            if not os.path.exists(directory):
                os.makedirs(directory)
            init_keys: Set[str] = set()
            super().__init__(init_keys, mode)
        elif mode is Mode.READ:
            if not os.path.isdir(directory):
                raise ValueError(directory + ' is not a directory')

            all_names = os.listdir(directory)
            png_names = [name for name in all_names if name.endswith('.png')]
            png_filenames = [name for name in png_names if
                             os.path.isfile(os.path.join(directory, name))]

            super().__init__(keys=set(png_filenames), mode=mode)

        self.__gray_transform = gray_transform

    def __getitem__(self, item: str) -> np.ndarray:
        super().__getitem__(item)
        image = cv2.imread(os.path.join(self.__dir, item))
        if self.__gray_transform:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[..., np.newaxis]
        return image

    def save_single(self, key: str, data: np.ndarray) -> None:
        """Saves one data entry to directory"""
        super().save_single(key, data)
        # self._keys.add(key)
        path = os.path.join(self.__dir, key)
        if self.__gray_transform:
            data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)[..., np.newaxis]
        cv2.imwrite(path, data)

    @classmethod
    def type(cls) -> str:
        """Returns type of this decorator"""
        return 'directory'
