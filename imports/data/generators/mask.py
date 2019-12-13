"""Keras generator for semantic segmentation tasks"""
import numpy as np
from tensorflow.python.keras.utils import Sequence

from imports.data.loaders.abstractloader import AbstractLoader


class MaskGenerator(Sequence):
    """Keras generator for semantic segmentation build on top of loaders"""

    def __init__(self, idx, loader, batch_size, augmentations=None, shuffle=True):
        """Constructor

        :param idx: indexes for loader
        :param loader: loader for masks and images
        :param batch_size: batch size
        :param augmentations: composed augmentations from albumentations package
        :param shuffle: shuffle data after every epoch
        """
        assert isinstance(loader, AbstractLoader)

        self.__loader = loader
        self.__batch_size = batch_size
        self.__indices = idx
        self.__shuffle = shuffle
        self.__augment = augmentations
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.ceil(len(self.__indices) / self.__batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        batch_idx = self.__indices[index * self.__batch_size:(index + 1) * self.__batch_size]

        images = [self.__loader.get_image(i) for i in batch_idx]
        masks = [self.__loader.get_mask(i) for i in batch_idx]
        if self.__augment is not None:
            data = [self.__augment(image=images[i], mask=masks[i]) for i in range(len(batch_idx))]
            images = [d['image'] for d in data]
            masks = [d['mask'] for d in data]
        return np.array(images), np.array(masks)

    def on_epoch_end(self):
        """This function is automatically called on epoch end"""
        if self.__shuffle:
            np.random.shuffle(self.__indices)
