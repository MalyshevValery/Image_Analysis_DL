import os
import numpy as np
import cv2


class ImageMaskLoader:
    """This class divides image data on train, validation, test sets and can create generators for model training"""

    def __init__(self, images_folder, masks_folder, train_val_test=(0.8, 0.1, 0.1), shuffle=True):
        """Constructor

        :param images_folder: Folder with images
        :param masks_folder: Folder with masks
        :param train_val_test: Fractures of train validation tests sets according to overall size
        :param shuffle: If data should be shuffled
        """
        assert len(train_val_test) == 3
        assert np.abs(np.sum(train_val_test) - 1) < 0.01

        self.__filenames = np.array(os.listdir(images_folder))
        self.__image_names = [os.path.join(images_folder, f) for f in self.__filenames]
        self.__mask_names = [os.path.join(masks_folder, f) for f in self.__filenames]
        self.__indices = np.arange(len(self.__filenames))
        if shuffle:
            np.random.shuffle(self.__indices)  # shuffle before split

        n_train = int(len(self.__filenames) * train_val_test[0])
        n_val = int(len(self.__filenames) * train_val_test[1])
        n_test = len(self.__filenames) - n_train - n_val

        self.__train = self.__indices[:n_train]
        self.__val = self.__indices[n_train:n_train + n_val]
        self.__test = self.__indices[-n_test:]

    def train_indices(self):
        return self.__train

    def valid_indices(self):
        return self.__val

    def test_indices(self):
        return self.__test

    def get_image(self, i):
        image = cv2.imread(self.__image_names[i]).astype(np.float32)
        return image

    def get_mask(self, i):
        """Returns (256,256) mask scaled to [0,1]"""
        mask = cv2.imread(self.__mask_names[i], cv2.IMREAD_GRAYSCALE).astype(np.float32)
        mask /= mask.max()
        return mask

    def save_predicted(self, directory, idx, pred):
        """Saves predicted masks

        :param directory: Dir where images will be saved
        :param idx: Indices of data entries
        :param pred: Predicted values
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

        pred = pred[:, :, :, 0]
        for i, val in enumerate(idx):
            image = cv2.imread(self.__image_names[val]) / 255.
            image = np.asarray(image, dtype=float) / np.max(image)
            image = cv2.resize(image, (256, 256), cv2.INTER_AREA)

            image[:, :, 2] *= (1.0 - pred[i])
            image = (255.0 * image).astype(np.uint8)

            pred_ = pred[i]
            pred_ = (255.0 * pred_).astype(np.uint8)

            cv2.imwrite(os.path.join(directory, self.__filenames[val]), image)
            cv2.imwrite(os.path.join(directory, self.__filenames[val] + '_mask.png'), pred_)
