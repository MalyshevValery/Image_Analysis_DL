import os
import numpy as np
import cv2


class ImageMaskGenerator:
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

        self.filenames = np.array(os.listdir(images_folder))
        self.image_names = [os.path.join(images_folder, f) for f in self.filenames]
        self.mask_names = [os.path.join(masks_folder, f) for f in self.filenames]
        self.indices = np.arange(len(self.filenames))
        if shuffle:
            np.random.shuffle(self.indices)

        n_train = int(len(self.filenames) * train_val_test[0])
        n_val = int(len(self.filenames) * train_val_test[1])
        n_test = len(self.filenames) - n_train - n_val

        self.train = self.indices[:n_train]
        self.val = self.indices[n_train:n_train + n_val]
        self.test = self.indices[-n_test:]

        self.batch_size = 1

    def set_batch_size(self, batch_size):
        """Sets batch size for generators"""
        assert batch_size >= 1
        self.batch_size = batch_size

    def train_steps(self):
        return self._batch_steps(self.train)

    def valid_steps(self):
        return self._batch_steps(self.val)

    def test_steps(self):
        return self._batch_steps(self.test)

    def train_generator(self):
        return self._data_generator(self.train)

    def valid_generator(self):
        return self._data_generator(self.val)

    def test_generator(self):
        return self._data_generator(self.test)

    def _batch_steps(self, array):
        return (len(array) + self.batch_size - 1) // self.batch_size

    def _read_one_batch(self, array_values):
        images = np.zeros((len(array_values), 256, 256, 3)).astype('float')
        masks = np.zeros((len(array_values), 256, 256, 1)).astype('float')

        for i, val in enumerate(array_values):
            image = cv2.imread(self.image_names[val]) / 255.
            image = np.asarray(image, dtype=float) / np.max(image)
            image = cv2.resize(image, (256, 256), cv2.INTER_AREA)

            mask = cv2.imread(self.mask_names[val], cv2.IMREAD_GRAYSCALE) / 255.
            mask = cv2.resize(mask, (256, 256), cv2.INTER_AREA) > 0.5
            mask = mask.reshape(256, 256, 1)  # Add extra dimension for parity with train_img size [512 * 512 * 3]

            images[i] = image  # add to array - img[0], img[1], and so on.
            masks[i] = mask
        return images, masks

    def _data_generator(self, array):
        while True:
            n_steps = self._batch_steps(array)
            for i in range(n_steps):
                delta = i * self.batch_size
                max_j = min(len(array) - delta, self.batch_size)

                yield self._read_one_batch(array[delta:delta + max_j])

    def save_predicted(self, directory, idx, pred):
        """Saves predicted masks

        :param directory: Dir where images will be saved
        :param idx: Indices of data entries (self.train, self.test, self.val)
        :param pred: Predicted values
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

        pred = pred[:, :, :, 0]
        for i, val in enumerate(idx):
            image = cv2.imread(self.image_names[val]) / 255.
            image = np.asarray(image, dtype=float) / np.max(image)
            image = cv2.resize(image, (256, 256), cv2.INTER_AREA)

            image[:, :, 2] *= (1.0 - pred[i])
            image = (255.0 * image).astype(np.uint8)

            cv2.imwrite(os.path.join(directory, self.filenames[val]), image)

