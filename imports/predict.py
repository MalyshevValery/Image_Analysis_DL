"""Predictor"""
import os

import cv2.cv2 as cv2
import numpy as np

import imports.utils as utils
from .data import OnlineGenerator


class PredictOnline:
    """Model predictor with online interface"""

    def __init__(self, jobdir):
        parser = utils.SettingsParser(os.path.join(jobdir, 'settings.json'), predict_mode=True)
        if 'load_gray' in parser.loader_args:
            self.load_gray = parser.loader_args['load_gray']
        else:
            self.load_gray = False
        self.generator = OnlineGenerator(parser.aug_all)

        self.model = parser.get_model_method()(parser.input_shape, **parser.model_params)
        self.model.load_weights(os.path.join(jobdir, 'weights.h5'))

    def __call__(self, image):
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
        print(image.shape, self.load_gray)
        if self.load_gray and image.shape[2] != 1:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)[:, :, np.newaxis]
        elif not self.load_gray and image.shape[2] == 1:
            image = np.repeat(image, axis=2, repeats=3)

        image_processed = self.generator.process_image(image)
        return self.model.predict(image_processed)[0]
