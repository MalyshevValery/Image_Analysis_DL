from tensorflow.python.keras.utils import Sequence
import numpy as np


class MaskGenerator(Sequence):
    def __init__(self, idx, loader, batch_size, augmentations=None, shuffle=True):
        """Keras generator for semantic segmentation tasks

        :param idx:
        :param loader:
        :param batch_size:
        :param augmentations:
        :param shuffle:
        """
        self.__loader = loader
        self.__batch_size = batch_size
        self.__idxs = idx
        self.__shuffle = shuffle
        self.__augment = augmentations
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.__idxs) / self.__batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        batch_idx = self.__idxs[index * self.__batch_size:(index + 1) * self.__batch_size]

        images = [self.__loader.get_image(i) for i in batch_idx]
        masks = [self.__loader.get_mask(i) for i in batch_idx]
        if self.__augment is not None:
            data = [self.__augment(image=images[i], mask=masks[i]) for i in range(len(batch_idx))]
            images = [d['image'] for d in data]
            masks = [d['mask'] for d in data]
        return np.array(images), np.array(masks)

    def on_epoch_end(self):
        if self.__shuffle:
            np.random.shuffle(self.__idxs)
