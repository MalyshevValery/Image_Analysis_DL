"""Online generator"""
import numpy as np


class OnlineGenerator:
    """Online generator for inference. It's useful when data is continuous flow and can not be divided on batches or
    static dataset."""

    def __init__(self, augmentations=None):
        """Constructor

        :param augmentations: composed augmentations from albumentations package
        """
        self.__augment = augmentations

    def process_image(self, image):
        """

        :param image: image
        :return: prepared image for neural network, which in a nutshell is just one item batch
        """
        data = self.__augment(image=image)
        image = data['image']
        return image[np.newaxis, ..., np.newaxis]
