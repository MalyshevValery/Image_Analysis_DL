"""Keras generator for semantic segmentation tasks"""
from typing import List, Tuple, Iterable

import numpy as np
from albumentations import BasicTransform
from tensorflow.keras.utils import Sequence

from .loader import Loader

_DataType = Tuple[Iterable[np.ndarray], Iterable[np.ndarray]]


class DataGenerator(Sequence):
    """Keras generator for loader and storages system.

    Can be used for large-scale static inference because it allows batches.
    """

    def __init__(self, keys: List[str], loader: Loader, batch_size: int = 1,
                 augmentations: BasicTransform = None, shuffle: bool = True):
        """Constructor

        :param keys: keys for data in loader
        :param loader: loader for masks and images
        :param batch_size: batch size
        :param augmentations: composed augmentations from albumentations package
        :param shuffle: shuffle data after every epoch
        """
        assert isinstance(loader, Loader)

        self._loader = loader
        self._batch_size = batch_size
        self._keys = keys
        self._shuffle = shuffle
        self._augment = augmentations
        self.on_epoch_end()

    def __len__(self) -> int:
        """Denotes the number of batches per epoch"""
        return int(np.ceil(len(self._keys) / self._batch_size))

    def __getitem__(self, index: int) -> _DataType:
        """Generate one batch of data"""
        batch_keys = self._keys[
                     index * self._batch_size:(index + 1) * self._batch_size]
        input_ = self._loader.get_input(batch_keys)
        output_ = self._loader.get_output(batch_keys)
        # TODO: augment
        return input_, output_

    def on_epoch_end(self) -> None:
        """This function is automatically called on epoch end"""
        if self._shuffle:
            np.random.shuffle(self._keys)

    @property
    def keys(self) -> List[str]:
        """Getter for keys"""
        return self._keys.copy()
