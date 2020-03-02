"""SplitMaskExtension tests"""
import unittest

import numpy as np

from imports.data.extensions import SplitMaskExtension


class SplitMaskTester(unittest.TestCase):
    """SplitMaskExtension tests"""

    def test_call(self) -> None:
        mask = np.array([
            [0, 1],
            [1, 0]
        ])
        true_val = np.array([
            [[1, 0], [0, 1]],
            [[0, 1], [1, 0]]
        ])
        ext = SplitMaskExtension()
        self.assertTrue(np.all(ext(mask) == true_val))

        mask[1][1] = 2
        true_val = np.array([
            [[1, 0, 0], [0, 1, 0]],
            [[0, 1, 0], [0, 0, 1]]
        ])
        ext = SplitMaskExtension(3)
        self.assertTrue(np.all(ext(mask) == true_val))

        true_val = np.array([
            [[1, 0], [0, 0]],
            [[0, 0], [0, 1]]
        ])
        ext = SplitMaskExtension([0, 2])
        self.assertTrue(np.all(ext(mask) == true_val))

    def test_exception(self) -> None:
        mask = np.array([
            [0, 1],
            [1, 0]
        ])[..., np.newaxis]
        ext = SplitMaskExtension()
        ext(mask)  # No exception
        self.assertRaises(ValueError,
                          lambda: ext(np.concatenate([mask, mask], axis=2)))

    def test_json(self) -> None:
        json_object = {
            'type': 'split_mask',
            'codes': [0, 1]
        }
        self.assertEqual(SplitMaskExtension().to_json(), json_object)

        json_object['codes'] = [0, 1, 2, 3]
        self.assertEqual(SplitMaskExtension(4).to_json(), json_object)

        json_object['codes'] = [0, 3]
        self.assertEqual(SplitMaskExtension([0, 3]).to_json(), json_object)
