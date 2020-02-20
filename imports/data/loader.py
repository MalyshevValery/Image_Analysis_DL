"""All-in-one loader for image tasks"""
from typing import List, Dict, Union

import numpy as np

from .extensions import AbstractExtension
from .overlay import image_mask
from .storages import AbstractStorage


class Loader:
    """Loader of data for different image tasks (currently only Semantic
    segmentation is supported)

    Loader is a middleware between storage and generators. It has two main
    tasks:
        1. Unite data from different storage to create bundles of input and
        ground truth data which can be passed to generators;

        2. Preprocess data by pushing it through specified extensions, which
        allows to change data types, scale it, add other effects.

    The further work on this class will be made in direction of adding other tasks to it.
    (bounding boxes, object detection and segmentation etc.). So this class will be configurable and flexible to provide
    any type of input data to generators.
    """

    def __init__(self, images: AbstractStorage = None, masks: AbstractStorage = None,
                 extensions: Dict[str, Union[AbstractExtension, List[AbstractExtension]]] = None):
        """Constructor

        :param images: Storage with images
        :param masks: Storage with masks
        :param extensions: dictionary of apply : list of extensions. Where apply is from [image, mask]
        """
        if images is not None and not isinstance(images, AbstractStorage):
            raise TypeError('Images have to be a Storage')

        self._extensions = {
            'image': [],
            'mask': [],
            'save_image': [],
            'save': []
        }
        if extensions is not None:
            apply_to = extensions.keys()
            for apply in apply_to:
                if apply not in self._extensions:
                    raise ValueError(apply, '- unknown apply to value')
                if not isinstance(extensions[apply], list):
                    extensions[apply] = [extensions[apply]]
                for extension in extensions[apply]:
                    if not isinstance(extension, AbstractExtension):
                        raise ValueError(extension + ' does not implements AbstractExtension')
                self._extensions[apply] = extensions[apply]

        if images is None:
            print('Loader prediction mode')
            return

        keys = images.keys
        self._images = images

        if masks is not None:
            if not isinstance(masks, AbstractStorage):
                raise TypeError('Masks have to be a Storage')
            keys = keys.intersection(masks.keys)
        self._masks = masks
        self._keys = list(keys)  # To ensure order

    def split(self, train_val_test):
        """Splits indices on three groups to create training, test and validation sets.

        :param train_val_test: tuple or list of three elements with sum of 1,
        which contains fractures of whole set for train/validation/test split
        :returns shuffled keys for train , validation, test split
        """
        if sum(train_val_test) > 1:
            raise ValueError('Split', train_val_test, 'is greater than 1')
        if len(train_val_test) != 3:
            raise ValueError('Split', train_val_test,
                             'must have fractures only for train, validation and test (3 elements)')
        np.random.shuffle(self._keys)
        train_val_test_counts = (np.array(train_val_test) * len(self._keys)).astype(int)
        train_count = train_val_test_counts[0]
        test_count = train_val_test_counts[2]
        return self._keys[:train_count], self._keys[train_count:-test_count], self._keys[-test_count:]

    def keys(self):
        """Getter for keys"""
        return self._keys

    def get_input_shape(self):
        """Returns input shape of data"""
        return self.get_image(self._keys[0]).shape

    def get_image(self, key):
        """Getter for input image"""
        image = self._images[key]
        for apply in self._extensions['image']:
            image = apply(image)
        return image

    def process_image(self, image):
        """Process external image"""
        for apply in self._extensions['image']:
            image = apply(image)
        return image

    def get_mask(self, key):
        """Get mask"""
        if self._masks is None:
            raise ValueError('Masks was not provided in constructor')
        mask = self._masks[key]
        for apply in self._extensions['mask']:
            mask = apply(mask)
        return mask

    def save_predicted(self, keys, predictions, storage: AbstractStorage):
        """Saves images with prediction overlay to storage.

        WARNING: currently works only for masks. In plans to add other types of ground truth data to it

        :param keys: keys of predictions
        :param predictions: array of predictions
        :param storage: storage to save to
        """
        if not isinstance(storage, AbstractStorage):
            raise TypeError('parameter storage is expected to be instance of Storage')
        images = np.array([self.get_image(key) for key in keys])
        for apply in self._extensions['save_image']:
            images = apply(images)
        array = image_mask(images, predictions)
        for apply in self._extensions['save']:
            array = apply(array)
            predictions = apply(predictions)
        storage.save_array(keys, array)

    def copy_for_storage(self, storage: AbstractStorage):
        """Creates new storage with other images storage to use it for predictions"""
        return Loader(images=storage, extensions=self._extensions)
