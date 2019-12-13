"""Image loader from folder"""
import numpy as np
import cv2.cv2 as cv2
import os

from imports.data.loaders.abstractloader import AbstractLoader


class ImageLoader(AbstractLoader):
    """This class provide images for prediction by loading them form specified folder"""

    def __init__(self, images_folder, load_gray=False):
        """Constructor

        :param images_folder: Folder with images
        :param load_gray: load images as gray or as RGB
        """

        self._filenames = np.array(os.listdir(images_folder))
        self._image_names = [os.path.join(images_folder, f) for f in self._filenames]

        self._indices = np.arange(len(self._filenames))
        self._load_gray = load_gray

    def get_indices(self):
        """Returns all indices of files"""
        return self._indices

    def get_image(self, i):
        """

        :param i: index of image
        :return:
        """
        if self._load_gray:
            return cv2.imread(self._image_names[i], cv2.IMREAD_GRAYSCALE)[:, :, np.newaxis]
        else:
            return cv2.imread(self._image_names[i])

    def get_mask(self, i):
        """Returns None because no mask existed for unpredicted images"""
        return None

    def save_predicted(self, directory, idx, pred):
        """Saves predicted masks to directory

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
