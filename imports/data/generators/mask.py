"""Keras generator for semantic segmentation tasks"""
import numpy as np

from .image import ImageGenerator


class MaskGenerator(ImageGenerator):
    """Keras generator for semantic segmentation build on top of loaders"""

    def __getitem__(self, index):
        """Generate one batch of data"""
        batch_keys = self._keys[index * self._batch_size:(index + 1) * self._batch_size]

        images = [self._loader.get_image(key) for key in batch_keys]
        masks = [self._loader.get_mask(key) for key in batch_keys]
        if self._augment is not None:
            data = [self._augment(image=images[i], mask=masks[i]) for i in range(len(batch_keys))]
            images = [d['image'] for d in data]
            masks = [d['mask'] for d in data]
        return np.array(images), np.array(masks)
