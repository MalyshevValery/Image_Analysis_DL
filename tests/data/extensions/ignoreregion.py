"""Ignore region Extension tests"""
import unittest

import numpy as np

from imports.data.extensions import IgnoreRegionExtension


class IgnoreRegionTester(unittest.TestCase):
    def setUp(self) -> None:
        self.__image = np.array([
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0],
        ], dtype=float)
        self.__split = np.stack([self.__image == 0, self.__image == 1], axis=2)
        self.__split = self.__split * 1.0

    def test_radius(self) -> None:
        ext = IgnoreRegionExtension()
        processed = ext(self.__split)
        self.assertTupleEqual(self.__split.shape, processed.shape)
        self.assertEqual(np.sum(processed), 0)

        ext = IgnoreRegionExtension(radius=1)
        test_image = np.array([
            [1, 0, 0, 0, 1],
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0],
            [1, 0, 0, 0, 1],
        ], dtype=float)
        self.assertTrue(np.all(test_image == np.sum(ext(self.__split), axis=2)))

        ext = IgnoreRegionExtension(radius=2)
        test_image *= 0
        test_image[2][2] = 1.0
        self.assertTrue(np.all(test_image == np.sum(ext(self.__split), axis=2)))

    def test_exception(self) -> None:
        ext = IgnoreRegionExtension()
        self.assertRaises(IndexError, lambda: ext(self.__split[0]))
        self.assertRaises(IndexError, lambda: ext(self.__split[..., 0]))
        doubled = np.concatenate([self.__split] * 2, axis=2)
        self.assertRaises(IndexError, lambda: ext(doubled))

    def test_json(self) -> None:
        json_object = {
            'type': 'ignore_region',
            'radius': 3,
        }
        self.assertEqual(json_object, IgnoreRegionExtension().to_json())

        json_object['radius'] = 5
        ext = IgnoreRegionExtension(radius=5)
        self.assertEqual(json_object, ext.to_json())
