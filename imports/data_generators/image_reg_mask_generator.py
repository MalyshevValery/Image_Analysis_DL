import os
import cv2
from tqdm import tqdm
import numpy as np
from imports.data_generators import ImageMaskGenerator
from imports.registration import Registration


class ImageRegMaskGenerator(ImageMaskGenerator):
    """Generator which appends registration mask to image"""
    def __init__(self, images_folder, masks_folder, reg_folder, descriptor_file,
                 train_val_test=(0.8, 0.1, 0.1), shuffle=True, **reg_args):
        """Constructor

        :param images_folder: Folder with images
        :param masks_folder: Folder with masks
        :param reg_folder: Folder for registration masks
        :param train_val_test: Fractures of train validation tests sets according to overall size
        :param shuffle: If data should be shuffled
        :param descriptor_file: File for descriptor (see registration params
        :param reg_args: Other keyword arguments for registration
        """

        assert len(train_val_test) == 3
        assert np.abs(np.sum(train_val_test) - 1) < 0.01
        self.filenames = np.array(os.listdir(images_folder))
        self.image_names = [os.path.join(images_folder, f) for f in self.filenames]
        self.mask_names = [os.path.join(masks_folder, f) for f in self.filenames]
        self.reg_names = [os.path.join(reg_folder, f) for f in self.filenames]
        self.indices = np.arange(len(self.filenames))
        if shuffle:
            np.random.shuffle(self.indices)

        n_train = int(len(self.filenames) * train_val_test[0])
        n_val = int(len(self.filenames) * train_val_test[1])
        n_test = len(self.filenames) - n_train - n_val

        self.train = self.indices[:n_train]
        self.val = self.indices[n_train:n_train + n_val]
        self.test = self.indices[-n_test:]

        if not os.path.isdir(reg_folder) or len(set(os.listdir(reg_folder)) & set(self.filenames)) < len(
                self.filenames):
            if not os.path.isdir(reg_folder):
                os.makedirs(reg_folder)

            reg = Registration(images_folder, masks_folder, descriptor_file, images_list=self.filenames[self.train],
                               forbid_the_same=True, **reg_args)
            for i in tqdm(range(len(self.filenames))):
                reg_mask = reg.segment(self.image_names[i])
                reg_mask *= 255.
                cv2.imwrite(self.reg_names[i], reg_mask)

        self.batch_size = 1

    def _read_one_batch(self, array_values):
        images = np.zeros((len(array_values), 256, 256, 2)).astype('float')
        masks = np.zeros((len(array_values), 256, 256, 1)).astype('float')

        for i, val in enumerate(array_values):
            image = cv2.imread(self.image_names[val], cv2.IMREAD_GRAYSCALE) / 255.
            image = np.asarray(image, dtype=float) / np.max(image)
            image = cv2.resize(image, (256, 256), cv2.INTER_AREA)

            reg_mask = cv2.imread(self.reg_names[val], cv2.IMREAD_GRAYSCALE) / 255.
            reg_mask = np.asarray(reg_mask, dtype=float) / np.max(reg_mask)
            reg_mask = cv2.resize(reg_mask, (256, 256), cv2.INTER_AREA)

            mask = cv2.imread(self.mask_names[val], cv2.IMREAD_GRAYSCALE) / 255.
            mask = cv2.resize(mask, (256, 256), cv2.INTER_AREA) > 0.5
            mask = mask.reshape(256, 256, 1)  # Add extra dimension for parity with train_img size [512 * 512 * 3]

            images[i, :, :, 0] = image  # add to array - img[0], img[1], and so on.
            images[i, :, :, 1] = reg_mask
            masks[i] = mask
        return images, masks
