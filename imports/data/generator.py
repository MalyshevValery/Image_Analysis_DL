"""Keras generator for semantic segmentation tasks"""
from typing import List, Union, Tuple, Dict

import numpy as np
from albumentations import BasicTransform, to_dict
from tensorflow.keras.utils import Sequence

from imports.utils.types import OMArray, to_seq
from .augmentationmap import AugmentationMap
from .loader import Loader


class DataGenerator(Sequence):
    """Keras generator for loader and storages system.

    Can be used for large-scale static inference because it allows batches.

    :param keys: Keys for data in loader
    :param loader: Loader for masks and images
    :param batch_size: Batch size
    :param aug_map: Composed augmentations from albumentations package
    :param transform: Albumentations transform to augment with it
    :param shuffle: Shuffle data after every epoch
    :param predict: True if generator is in predict mode
    """

    def __init__(self, keys: List[str], loader: Loader,
                 batch_size: int = 1, aug_map: AugmentationMap = None,
                 transform: BasicTransform = None, shuffle: bool = True,
                 predict: bool = False):
        self._loader = loader
        self._batch_size = batch_size
        self._keys = np.array(keys)
        self._shuffle = shuffle
        self._augment = aug_map
        self._transform = transform
        self._predict = predict
        self.on_epoch_end()

    def __len__(self) -> int:
        """Denotes the number of batches per epoch"""
        return int(np.ceil(len(self._keys) / self._batch_size))

    def __getitem__(self,
                    index: int) -> Union[OMArray, Tuple[OMArray, OMArray]]:
        """Generate one batch of data"""
        batch_keys = self._keys[
                     index * self._batch_size:(index + 1) * self._batch_size]
        input_ = to_seq(self._loader.get_input(batch_keys))
        if self._predict:
            return input_

        output_ = to_seq(self._loader.get_output(batch_keys))
        if self._augment is not None and self._transform is not None:
            return self._augment(input_, output_, self._transform)
        return input_, output_

    def on_epoch_end(self) -> None:
        """This function is automatically called on epoch end"""
        if self._shuffle:
            np.random.shuffle(self._keys)

    @property
    def keys(self) -> List[str]:
        """Getter for keys"""
        return self._keys.copy()

    def to_json(self) -> Dict[str, object]:
        """Returns JSON config for this object"""
        transform_json = to_dict(self._transform) if self._transform else None
        return {
            'batch_size': self._batch_size,
            'shuffle': self._shuffle,
            'predict': self._predict,
            'n_keys': len(self._keys),
            'aug_map': self._augment.to_json() if self._augment else None,
            'augmentation': transform_json,
            'keys': self._keys.tolist(),
        }
