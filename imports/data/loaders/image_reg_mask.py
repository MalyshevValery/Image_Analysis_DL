"""Loader for training with additional registration mask"""
import os
import shutil
import sys

import cv2.cv2 as cv2
import numpy as np
from tqdm import tqdm

from imports.registration import Registration

print('ImageRegMaskLoader is not supported and is waiting for remake for new system', file=sys.stderr)


class ImageRegMaskLoader():
    """Generator which appends registration mask to image"""

    def __init__(self, images_folder, masks_folder, reg_folder, descriptor_file,
                 train_val_test=(0.8, 0.1, 0.1), shuffle=True, load_gray=False, mask_channel_codes=None,
                 delete_previous=False, **reg_args):
        """Constructor

        :param images_folder: Folder with images
        :param masks_folder: Folder with masks
        :param reg_folder: Folder for registration masks
        :param train_val_test: Fractures of train validation tests sets according to overall size
        :param shuffle: If data should be shuffled
        :param descriptor_file: File for descriptor (see registration params
        :param reg_args: Other keyword arguments for registration
        """
        super().__init__(images_folder, masks_folder, train_val_test, shuffle, load_gray, mask_channel_codes)
        self._reg_names = [os.path.join(reg_folder, f) for f in self._filenames]
        if os.path.isdir(reg_folder) and delete_previous:
            shutil.rmtree(reg_folder)

        if not os.path.isdir(reg_folder) or len(set(os.listdir(reg_folder)) & set(self._filenames)) < len(
                self._filenames):
            if not os.path.isdir(reg_folder):
                os.makedirs(reg_folder)

            reg = Registration(images_folder, masks_folder, descriptor_file,
                               images_list=self._filenames[super().train_indices()], forbid_the_same=True, **reg_args)
            for i in tqdm(range(len(self._filenames))):
                reg_mask = reg.segment(self._image_names[i])
                reg_mask *= 255.
                cv2.imwrite(self._reg_names[i], reg_mask)

    def get_image(self, i):
        """Returns image with additional channel, representing mask acquired by registration

        :param i: index of image
        :return: image
        """
        if self._load_gray:
            image = cv2.imread(self._image_names[i], cv2.IMREAD_GRAYSCALE)[:, :, np.newaxis]
        else:
            image = cv2.imread(self._image_names[i])
        reg_mask = cv2.imread(self._reg_names[i], cv2.IMREAD_GRAYSCALE)[:, :, np.newaxis]
        return np.concatenate([image, reg_mask], axis=2)
