"""TypeScaleExtension tests"""
import unittest

import numpy as np

from imports.data.extensions import TypeScaleExtension


class TypeScaleTester(unittest.TestCase):
    """Type Scale tests"""

    def test_call(self) -> None:
        data = np.array([0.1, 0.5, 0.75])
        ext = TypeScaleExtension(1, np.uint8, 255)
        self.assertTrue(np.all(ext(data) == np.array([25, 127, 191])))

        ext = TypeScaleExtension(1, np.uint8)
        self.assertTrue(np.all(ext(data) == np.array([25, 127, 191])))

        ext = TypeScaleExtension(target_type=np.uint8)
        self.assertTrue(np.all(ext(data) == np.array([34, 170, 255])))

    def test_to_json(self) -> None:
        ext = TypeScaleExtension(1, np.uint8, 255)
        expected = {
            'type': 'type_scale',
            'target_type': 'uint8',
            'src_max': 1,
            'target_max': 255
        }
        self.assertEqual(ext.to_json(), expected)

        ext = TypeScaleExtension(1, np.uint8)
        self.assertEqual(ext.to_json(), expected)

        ext = TypeScaleExtension(target_type=np.uint8)
        expected['src_max'] = None
        self.assertEqual(ext.to_json(), expected)
