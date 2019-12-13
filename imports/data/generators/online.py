"""Online generator"""
import numpy as np


class OnlineGenerator:
    """Online generator for making predictions"""

    def __init__(self, augmentations=None):
        """Constructor

        :param augmentations: composed augmentations from albumentations package
        """
        self.__augment = augmentations

    def process_image(self, image):
        """

        :param image: image
        :return: prepared image for NN
        """
        data = self.__augment(image=image)
        image = data['image']
        return image[np.newaxis]
