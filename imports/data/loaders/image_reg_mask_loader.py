import os
import shutil

import cv2
from tqdm import tqdm
import numpy as np
from imports.data.loaders.image_mask_loader import ImageMaskLoader
from imports.registration import Registration


class ImageRegMaskLoader(ImageMaskLoader):
    """Generator which appends registration mask to image"""

    def __init__(self, images_folder, masks_folder, reg_folder, descriptor_file,
                 train_val_test=(0.8, 0.1, 0.1), shuffle=True, delete_previous=False, **reg_args):
        """Constructor

        :param images_folder: Folder with images
        :param masks_folder: Folder with masks
        :param reg_folder: Folder for registration masks
        :param train_val_test: Fractures of train validation tests sets according to overall size
        :param shuffle: If data should be shuffled
        :param descriptor_file: File for descriptor (see registration params
        :param reg_args: Other keyword arguments for registration
        """
        super().__init__(images_folder, masks_folder, train_val_test, shuffle)
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
        image = cv2.imread(self._image_names[i], cv2.IMREAD_GRAYSCALE).astype(np.float32)
        reg_mask = cv2.imread(self._reg_names[i], cv2.IMREAD_GRAYSCALE).astype(np.float32)
        return np.stack([image, reg_mask], axis=-1)