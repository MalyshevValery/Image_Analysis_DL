import os
import numpy as np
import cv2


class DataGenerator:
    def __init__(self, images_folder, masks_folder, train_val_test=(0.8, 0.1, 0.1), shuffle=True):
        assert len(train_val_test) == 3
        assert np.abs(np.sum(train_val_test) - 1) < 0.01

        self.names = os.listdir(images_folder)
        self.image_names = [os.path.join(images_folder, f) for f in self.image_names]
        self.mask_names = [os.path.join(masks_folder, f) for f in self.image_names]
        self.indices = np.arange(len(self.names))
        if shuffle:
            np.random.shuffle(self.indices)

        n_train = int(len(self.names) * train_val_test[0])
        n_val = int(len(self.names) * train_val_test[1])
        n_test = len(self.names) - n_train - n_val

        self.train = self.indices[:n_train]
        self.val = self.indices[n_train:n_train + n_val]
        self.test = self.indices[-n_test:]

    def train_steps(self):
        return self.__batch_steps(self.train)

    def __batch_steps(self, array, batch_size=1):
        if batch_size is None:
            batch_size = 1
        return (len(array) + batch_size - 1) // batch_size

    def __data_generator(self, array, batch_size=1):
        n_steps = self.__batch_steps(array, batch_size)
        for i in range(n_steps):

            images = np.zeros((batch_size, 256, 256, 3)).astype('float')
            masks = np.zeros((batch_size, 256, 256, 1)).astype('float')

            delta = i * batch_size
            max_j = min(len(array) - delta, batch_size)
            for j in range(0, max_j):
                image = cv2.imread(self.image_names[array[j + delta]]) / 255.
                image = np.asarray(image, dtype=float) / np.max(image)
                image = cv2.resize(image, (256, 256), cv2.INTER_AREA)

                mask = cv2.imread(self.mask_names[array[j + delta]], cv2.IMREAD_GRAYSCALE) / 255.
                mask = cv2.resize(mask, (256, 256), cv2.INTER_AREA) > 0.5
                mask = mask.reshape(256, 256, 1)  # Add extra dimension for parity with train_img size [512 * 512 * 3]

                images[j] = image  # add to array - img[0], img[1], and so on.
                masks[j] = mask

            if i == n_steps - 1:
                np.random.shuffle(array)
            yield images, masks
