"""Tests for composed storage"""
import os
import unittest
from shutil import rmtree

import numpy as np
from skimage.color import rgb2gray
from skimage.io import imread, imsave

from imports.data.extensions import TypeScaleExtension
from imports.data.storages import DirectoryStorage


class DirectoryStorageTester(unittest.TestCase):
    def setUp(self) -> None:
        """Create folder with test files"""
        self._dir = 'test_dir'
        self._n = 5
        os.mkdir(self._dir)
        for i in range(self._n):
            imsave(f'{self._dir}/{i}.png',
                   np.random.randint(0, 255, (256, 256, 3)))

    def tearDown(self) -> None:
        """Remove files"""
        rmtree(self._dir)

    def test_read(self) -> None:
        type_scale = TypeScaleExtension(src_max=255, target_type=np.float32,
                                        target_max=1.0)
        storage = DirectoryStorage(self._dir, extensions=type_scale)
        self.assertEqual(len(storage), self._n)
        for i in range(self._n):
            true_image = imread(f'{self._dir}/{i}.png') / 255
            diff = np.mean(np.abs(true_image - storage[f'{i}.png']))
            self.assertAlmostEqual(float(diff), 0.0, 6)
        self.assertRaises(ValueError,
                          lambda: storage.save_single('-', np.array(1)))

    def test_gray(self) -> None:
        storage = DirectoryStorage(self._dir, gray=True)
        true_image = imread(f'{self._dir}/0.png') / 255
        diff = np.mean(np.abs(rgb2gray(true_image) - storage['0.png'][..., 0]))
        self.assertAlmostEqual(float(diff), 0, 6)

    def test_write(self) -> None:
        storage = DirectoryStorage(self._dir, writable=True)
        self.assertEqual(len(storage), self._n)
        images = []
        for i in range(3):
            images.append(np.random.randint(0, 255, (256, 256, 3)))
        images = np.array(images)

        storage.save_array(['t1.png', 't2.png'], images[:2])
        storage.save_single('t3.png', images[2])
        self.assertEqual(len(storage), self._n + 3)
        self.assertEqual(len(os.listdir(self._dir)), self._n + 3)
        self.assertTrue(np.all(imread(f'{self._dir}/t2.png') == images[1]))

    def test_json(self) -> None:
        type_scale = TypeScaleExtension(src_max=255, target_type=np.float32,
                                        target_max=1.0)
        storage = DirectoryStorage(self._dir, extensions=type_scale, gray=True)
        json_ret = {
            'type': 'directory',
            'gray': True,
            'directory': self._dir,
            'extensions': [type_scale.to_json()]
        }
        self.assertEqual(storage.to_json(), json_ret)
