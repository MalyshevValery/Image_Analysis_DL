"""Keras generator for semantic segmentation tasks"""
import numpy as np
from albumentations import BasicTransform
from tensorflow.python.keras.utils import Sequence

from imports.data.loader import Loader


class MaskGenerator(Sequence):
    """Keras generator for semantic segmentation build on top of loaders"""

    def __init__(self, keys, loader: Loader, batch_size, augmentations: BasicTransform = None, shuffle=True):
        """Constructor

        :param keys: keys for data in loader
        :param loader: loader for masks and images
        :param batch_size: batch size
        :param augmentations: composed augmentations from albumentations package
        :param shuffle: shuffle data after every epoch
        """
        assert isinstance(loader, Loader)

        self.__loader = loader
        self.__batch_size = batch_size
        self.__keys = keys
        self.__shuffle = shuffle
        self.__augment = augmentations
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.ceil(len(self.__keys) / self.__batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        batch_keys = self.__keys[index * self.__batch_size:(index + 1) * self.__batch_size]

        images = [self.__loader.get_image(key) for key in batch_keys]
        masks = [self.__loader.get_mask(key) for key in batch_keys]
        if self.__augment is not None:
            data = [self.__augment(image=images[i], mask=masks[i]) for i in range(len(batch_keys))]
            images = [d['image'] for d in data]
            masks = [d['mask'] for d in data]
        return np.array(images), np.array(masks)

    def on_epoch_end(self):
        """This function is automatically called on epoch end"""
        if self.__shuffle:
            np.random.shuffle(self.__keys)

    @property
    def keys(self):
        """Getter for keys"""
        return self.__keys.copy()
