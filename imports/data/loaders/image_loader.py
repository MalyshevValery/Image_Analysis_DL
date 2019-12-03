import numpy as np
import cv2
import os

from imports.data.loaders import ImageMaskLoader


class ImageLoader(ImageMaskLoader):
    """This class provide images for prediction"""

    def __init__(self, images_folder, frac=1, load_gray=False, mask_channel_codes=None):
        """Constructor

        :param images_folder: Folder with images
        """
        super(ImageLoader, self).__init__(images_folder, None, (0, 1 - frac, frac), True, load_gray=load_gray,
                                          mask_channel_codes=mask_channel_codes)

    def get_mask(self, i):
        """Returns (256,256) mask scaled to [0,1]"""
        shape3 = 1
        if self._mask_channel_codes is not None:
            shape3 = len(self._mask_channel_codes)
        return np.zeros((256, 256, shape3), dtype=np.float32)