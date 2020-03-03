"""Tests for composed storage"""
import unittest

import numpy as np

from imports.data.extensions import TypeScaleExtension
from imports.data.storages import ClassKeyStorage


class ClassKeyStorageTester(unittest.TestCase):
    def test_read(self) -> None:
        type_scale = TypeScaleExtension(src_max=2, target_max=100)
        storage = ClassKeyStorage({'11', '22', '23', '33'}, '\\d(\\d)',
                                  extensions=type_scale)
        self.assertEqual(len(storage), 4)
        self.assertEqual(len(storage.class_dict), 3)
        self.assertEqual(storage['01'], 0)
        self.assertEqual(storage['13'], 100)
        self.assertRaises(NotImplementedError,
                          lambda: storage.save_single('-', np.array(1)))

    def test_one_hot(self) -> None:
        type_scale = TypeScaleExtension(src_max=2, target_max=100)
        storage = ClassKeyStorage({'11', '22', '23', '33'}, '\\d(\\d)',
                                  extensions=type_scale, one_hot=True)
        self.assertTrue(np.all(storage['01'] == [50, 0, 0]))
        self.assertTrue(np.all(storage['13'] == [0, 0, 50]))

    def test_write(self) -> None:
        self.assertRaises(ValueError,
                          lambda: ClassKeyStorage({'11', '22', '23', '33'},
                                                  '\\d(\\d)',
                                                  writable=True))

    def test_json(self) -> None:
        type_scale = TypeScaleExtension(src_max=4, target_max=100)
        storage = ClassKeyStorage({'11', '22', '23', '33'}, '\\d(\\d)',
                                  extensions=type_scale)
        json_ret = {
            'type': 'class_key',
            'pattern': '\\d(\\d)',
            'class_dict': storage.class_dict,
            'extensions': [type_scale.to_json()]
        }
        self.assertEqual(storage.to_json(), json_ret)
