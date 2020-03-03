"""Mock storage tests"""
import unittest
from typing import Set

import numpy as np

from imports.data.extensions.typescale import TypeScaleExtension
from imports.data.storages import MockStorage


class MockStorageTester(unittest.TestCase):
    def test_read(self) -> None:
        ms = MockStorage(np.array(1.0), {'1', '2'},
                         extensions=TypeScaleExtension())
        self.assertEqual(len(ms), 2)
        self.assertEqual(ms['1'], 255)
        self.assertEqual(ms['2'], 255)
        self.assertRaises(ValueError, lambda: ms.save_single('-', np.array(-1)))

    def test_write(self) -> None:
        keys: Set[str] = set()
        ms = MockStorage(np.array(1.0), keys, extensions=TypeScaleExtension(),
                         writable=True)
        self.assertEqual(len(ms), 0)
        ms.save_array(['1', '2'], np.array([1, 2]))
        ms.save_single('3', np.array([1, 2]))
        self.assertEqual(len(ms), 3)
        self.assertEqual(ms.keys, {'1', '2', '3'})

    def test_json(self) -> None:
        ms = MockStorage(np.array(1.0), {'1', '2'},
                         extensions=TypeScaleExtension())
        json_ret = {
            'type': 'mock',
            'val': 1.0,
            'extensions': [TypeScaleExtension().to_json()]
        }
        self.assertEqual(ms.to_json(), json_ret)
