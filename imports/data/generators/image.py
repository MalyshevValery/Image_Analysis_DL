"""Keras generator for inference"""
import numpy as np
from albumentations import BasicTransform
from tensorflow.python.keras.utils import Sequence

from imports.data.loader import Loader


class ImageGenerator(Sequence):
    """Keras generator for inference by providing only images"""

    def __init__(self, keys, loader: Loader, batch_size, augmentations: BasicTransform = None, shuffle=True):
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

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.ceil(len(self._keys) / self._batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        batch_keys = self._keys[index * self._batch_size:(index + 1) * self._batch_size]

        images = [self._loader.get_image(key) for key in batch_keys]
        if self._augment is not None:
            data = [self._augment(image=images[i]) for i in range(len(batch_keys))]
            images = [d['image'] for d in data]
        return np.array(images)

    def on_epoch_end(self):
        """This function is automatically called on epoch end"""
        if self._shuffle:
            np.random.shuffle(self._keys)

    @property
    def keys(self):
        """Getter for keys"""
        return self._keys.copy()
