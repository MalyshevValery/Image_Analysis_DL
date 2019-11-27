import os
import numpy as np
import cv2


class ImageMaskLoader:
    """This class divides image data on train, validation, test sets and can create generators for model training"""

    def __init__(self, images_folder, masks_folder, train_val_test=(0.8, 0.1, 0.1), shuffle=True, load_gray=False,
                 mask_channel_codes=None):
        """Constructor

        :param images_folder: Folder with images
        :param masks_folder: Folder with masks
        :param train_val_test: Fractures of train validation tests sets according to overall size
        :param shuffle: If data should be shuffled
        """
        assert len(train_val_test) == 3
        assert np.abs(np.sum(train_val_test) - 1) < 0.01

        self._filenames = np.array(os.listdir(images_folder))
        self._image_names = [os.path.join(images_folder, f) for f in self._filenames]
        self._mask_names = [os.path.join(masks_folder, f) for f in self._filenames]
        self._indices = np.arange(len(self._filenames))
        self._load_gray = load_gray
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
        return self.__train

    def valid_indices(self):
        return self.__val

    def test_indices(self):
        return self.__test

    def get_image(self, i):
        if self._load_gray:
            return cv2.imread(self._image_names[i], cv2.IMREAD_GRAYSCALE)[:, :, np.newaxis]
        else:
            return cv2.imread(self._image_names[i])

    def get_mask(self, i):
        """Returns (256,256) mask scaled to [0,1]"""
        mask = cv2.imread(self._mask_names[i], cv2.IMREAD_GRAYSCALE).astype(np.float32)
        mask /= mask.max()
        if self._mask_channel_codes is None:
            return mask[:, :, np.newaxis]
        else:
            mask *= max(self._mask_channel_codes)
            masks = [np.abs(mask - c) < 0.01 for c in self._mask_channel_codes]
            mask = np.stack(masks, axis=-1)
            return mask.astype(np.float32)

    def save_predicted(self, directory, idx, pred):
        """Saves predicted masks

        :param directory: Dir where images will be saved
        :param idx: Indices of data entries
        :param pred: Predicted values
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

        for i, val in enumerate(idx):
            image = cv2.imread(self._image_names[val]) / 255.
            image = np.asarray(image, dtype=float) / np.max(image)
            image = cv2.resize(image, (256, 256), cv2.INTER_AREA)

            image[:, :, :pred[i].shape[-1]] *= (1.0 - pred[i])
            image = (255.0 * image).astype(np.uint8)

            pred_ = pred[i]
            pred_ = (255.0 * pred_).astype(np.uint8)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(directory, self._filenames[val]), image)
            for j in range(pred_.shape[-1]):
                cv2.imwrite(os.path.join(directory, self._filenames[val] + '_mask{}.png'.format(j)), pred_[:, :, j])
