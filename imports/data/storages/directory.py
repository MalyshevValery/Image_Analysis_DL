"""Directory storage"""
import os
from typing import Dict

import numpy as np
import skimage.io as io
import skimage.util as util

from .abstract import AbstractStorage, ExtensionType


class DirectoryStorage(AbstractStorage):
    """Directory storage that loads all **PNG** images in directory

    :param directory: directory with PNG images
    :param gray: True to transform images to grayscale after reading
        and before writing. **NOTE:** when using this param return value will be
        [0, 1] float64
    :param writable: True to allow writing into directory
    :param extensions: Extensions to apply to this storage
    """

    def __init__(self, directory: str, gray: bool = False,
                 extensions: ExtensionType = None, writable: bool = False):
        self.__dir = directory
        self.__gray = gray

        if writable and not os.path.exists(directory):
            if not os.path.exists(directory):
                os.makedirs(directory)
        elif not os.path.isdir(directory):
            raise ValueError(f'{directory} is not a directory')

        all_names = os.listdir(directory)
        png_names = [name for name in all_names if name.endswith('.png')]
        png_filenames = [name for name in png_names if
                         os.path.isfile(os.path.join(directory, name))]
        keys = set(png_filenames)

        super().__init__(keys, extensions, writable)

    def __getitem__(self, item: str) -> np.ndarray:
        image = io.imread(os.path.join(self.__dir, item), as_gray=self.__gray)
        if self.__gray:
            image = image[..., np.newaxis]
        return self._apply_extensions(image)

    def save_single(self, key: str, data: np.ndarray) -> None:
        """Saves one data entry to directory"""
        if not self.writable:
            raise ValueError("Not writable")
        self._add_keys(key)
        path = os.path.join(self.__dir, key)
        io.imsave(path, util.img_as_ubyte(data))

    def to_json(self) -> Dict[str, object]:
        """Returns JSON configuration for this Storage"""
        return {
            'type': 'directory',
            'directory': self.__dir,
            'gray': self.__gray,
            'extensions': self._extensions_json()
        }
