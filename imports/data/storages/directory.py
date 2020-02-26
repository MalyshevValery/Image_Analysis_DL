"""Directory storage"""
import os
from typing import Set

import numpy as np
import skimage.color as color
import skimage.io as io

from .abstract import AbstractStorage, Mode, ExtensionType


class DirectoryStorage(AbstractStorage):
    """Directory storage that loads all **PNG** images in directory

    :param directory: directory with PNG images
    :param gray: True to transform images to grayscale after reading
        and before writing
    :param mode: Mode.READ for read Mode.WRITE for write
    :param extensions: Extensions to apply to this storage
    """

    def __init__(self, directory: str, gray: bool = False,
                 mode: Mode = Mode.READ, extensions: ExtensionType = None):
        self.__dir = directory
        self.__gray = gray

        if mode is Mode.WRITE:
            if not os.path.exists(directory):
                os.makedirs(directory)
            init_keys: Set[str] = set()
            keys = init_keys
        elif mode is Mode.READ:
            if not os.path.isdir(directory):
                raise ValueError(directory + ' is not a directory')

            all_names = os.listdir(directory)
            png_names = [name for name in all_names if name.endswith('.png')]
            png_filenames = [name for name in png_names if
                             os.path.isfile(os.path.join(directory, name))]
            keys = set(png_filenames)
        else:
            raise ValueError('Wrong Mode')

        super().__init__(keys, mode, extensions)

    def __getitem__(self, item: str) -> np.ndarray:
        super().__getitem__(item)
        image = io.imread(os.path.join(self.__dir, item), as_gray=self.__gray)
        if self.__gray:
            image = image[..., np.newaxis]
        return self._apply_extensions(image)

    def save_single(self, key: str, data: np.ndarray) -> None:
        """Saves one data entry to directory"""
        super().save_single(key, data)
        # self._keys.add(key)
        path = os.path.join(self.__dir, key)
        if self.__gray:
            data = color.rgb2gray(data)
        io.imsave(path, data)

    @classmethod
    def type(cls) -> str:
        """Returns type of this decorator"""
        return 'directory'
