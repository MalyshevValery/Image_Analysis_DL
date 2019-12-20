"""All-in-one loader for image tasks"""
import copy

import numpy as np

from imports.data.extensions import extension_factory, AbstractExtension
from imports.data.overlay import image_mask
from imports.data.storages import AbstractStorage, storage_factory
from imports.utils.jsonserializable import JSONSerializable


class Loader(JSONSerializable):
    """Loader of data for different image tasks (currently only Semantic segmentation is supported)"""

    def __init__(self, images, masks=None, extensions=None):
        """Constructor

        :param images: Storage with images
        :param masks: Storage with masks
        :param extensions: dictionary of apply : list of extensions. Where apply is from [image, mask]
        """
        if not isinstance(images, AbstractStorage):
            raise TypeError('Images have to be a Storage')

        keys = images.get_keys()
        self._images = images

        if masks is not None:
            if not isinstance(masks, AbstractStorage):
                raise TypeError('Masks have to be a Storage')
            keys = keys.intersection(masks.get_keys())
        self._masks = masks

        self._keys = list(keys)  # To ensure order

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
                    extension.check_extension(apply)
                self._extensions[apply] = extensions[apply]

    def split(self, train_val_test):
        """Splits indices on three fractures

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

    def get_keys(self):
        """Getter for keys"""
        return self._keys

    def get_input_shape(self):
        """Returns input shape of data"""
        return self.get_image(self._keys[0]).shape

    def get_image(self, key):
        """Get image"""
        image = self._images[key]
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

    def save_predicted(self, keys, predictions, storage):
        """Saves images with prediction overlay to storage.

        WARNING: currently works only for masks

        :param keys: keys of predictions
        :param predictions: array of predictions
        :param storage: storage to save to
        :return:
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

    def to_json(self):
        """Returns JSON representation of Loader"""
        json = {'images': self._images.to_json()}
        if self._masks is not None:
            json['masks'] = self._masks.to_json()
        if self._extensions is not None:
            to_save = {}
            for k in self._extensions.keys():
                to_save[k] = [extension.to_json() for extension in self._extensions[k]]
            json['extensions'] = to_save
        return json

    @staticmethod
    def from_json(json):
        """Creates loader from JSON"""
        config = copy.deepcopy(json)
        del config['images']
        for k in config.keys():
            if k == 'extensions':
                extensions = {}
                extensions_conf = config['extensions']
                for apply in extensions_conf:
                    if not isinstance(extensions_conf[apply], list):
                        extensions[apply] = extension_factory(extensions_conf[apply], apply)
                    else:
                        new_extensions = []
                        for extension in extensions_conf[apply]:
                            new_extensions.append(extension_factory(extension, apply))
                        extensions[apply] = new_extensions
                config[k] = extensions
            else:
                config[k] = storage_factory(config[k])
        return Loader(storage_factory(json['images']), **config)
