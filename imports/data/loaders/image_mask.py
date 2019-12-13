"""Loader of images and masks from folder to use during training process"""
import os
import numpy as np
import cv2.cv2 as cv2
from imports.data.loaders.image import ImageLoader
# TODO: More detailed explanation of mask_channel_codes


class ImageMaskLoader(ImageLoader):
    """This class divides image data on train, validation, test sets and can create generators for model training"""

    def __init__(self, images_folder, masks_folder, train_val_test=(0.8, 0.1, 0.1), shuffle=True, load_gray=False,
                 mask_channel_codes=None):
        """Constructor

        :param images_folder: Folder with images
        :param masks_folder: Folder with masks
        :param train_val_test: Fractures of train validation tests sets according to overall size
        :param shuffle: If data should be shuffled
        :param load_gray: True to load image as 1 channel grayscale
        :param mask_channel_codes: None if mask must have 1 channel, int to define number of channels in mask.
        """
        assert len(train_val_test) == 3
        assert np.abs(np.sum(train_val_test) - 1) < 0.01
        super().__init__(images_folder, load_gray)

        self._mask_names = [os.path.join(masks_folder, f) for f in self._filenames]
        self._mask_channel_codes = mask_channel_codes

        if isinstance(self._mask_channel_codes, int):
            self._mask_channel_codes = list(range(self._mask_channel_codes))
        if shuffle:
            np.random.shuffle(self._indices)  # shuffle before split

        n_train = int(len(self._filenames) * train_val_test[0])
        n_val = int(len(self._filenames) * train_val_test[1])
        n_test = len(self._filenames) - n_train - n_val

        self.__train = self._indices[:n_train]
        self.__val = self._indices[n_train:n_train + n_val]
        self.__test = self._indices[-n_test:]

    def train_indices(self):
        """Returns train indices"""
        return self.__train

    def valid_indices(self):
        """Returns validation set indices"""
        return self.__val

    def test_indices(self):
        """Returns train set indices"""
        return self.__test

    def get_mask(self, i):
        """Returns (256,256) mask scaled to [0,1]

        :param i: index of image
        :return: image mask scaled to 0,1.
        """
        mask = cv2.imread(self._mask_names[i], cv2.IMREAD_GRAYSCALE).astype(np.float32)
        mask /= mask.max()
        if self._mask_channel_codes is None:
            return mask[:, :, np.newaxis]
        else:
            mask *= max(self._mask_channel_codes)
            masks = [np.abs(mask - c) < 0.01 for c in self._mask_channel_codes]
            mask = np.stack(masks, axis=-1)
            return mask.astype(np.float32)
