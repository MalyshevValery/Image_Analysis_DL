"""Tests for composed storage"""
import unittest
from typing import Tuple

import numpy as np

from imports.data.extensions import TypeScaleExtension
from imports.data.storages import ComposeStorage, MockStorage

_MockType = Tuple[MockStorage, MockStorage]


class ComposeStorageTester(unittest.TestCase):
    @staticmethod
    def get_mock_storages(writable: bool = False) -> _MockType:
        """Returns two mock storages with two keys each"""
        ms1 = MockStorage(np.array(1.0), {'1', '2'}, writable=writable)
        ms2 = MockStorage(np.array(2.0), {'3', '4'}, writable=writable)
        return ms1, ms2

    def test_read(self) -> None:
        storage = ComposeStorage(ComposeStorageTester.get_mock_storages(),
                                 extensions=TypeScaleExtension(src_max=2))
        self.assertEqual(len(storage), 4)
        self.assertEqual(storage['0/1'], 127)
        self.assertEqual(storage['1/3'], 255)
        self.assertRaises(ValueError, lambda: storage['-'])
        self.assertRaises(ValueError,
                          lambda: storage.save_single('-', np.array(1)))

    def test_writable(self) -> None:
        mss = ComposeStorageTester.get_mock_storages()
        self.assertRaises(ValueError,
                          lambda: ComposeStorage(mss, writable=True))

    def test_write(self) -> None:
        mss = ComposeStorageTester.get_mock_storages(True)
        storage = ComposeStorage(mss, extensions=TypeScaleExtension(src_max=2),
                                 writable=True)
        self.assertEqual(len(storage), 4)
        storage.save_array(['0/0', '1/2'], np.array([1, 2]))
        storage.save_single('1/5', np.array([1, 2]))
        self.assertEqual(len(storage), 7)
        self.assertEqual(storage.keys,
                         {'0/0', '0/1', '0/2', '1/2', '1/3', '1/4', '1/5'})
        self.assertEqual(len(mss[0]), 3)
        self.assertEqual(len(mss[1]), 4)
        self.assertEqual(mss[0].keys, {'0', '1', '2'})
        self.assertEqual(mss[1].keys, {'2', '3', '4', '5'})

    def test_json(self) -> None:
        mss = ComposeStorageTester.get_mock_storages()
        storage = ComposeStorage(mss, extensions=TypeScaleExtension())
        json_ret = {
            'type': 'compose',
            'storages': [ms.to_json() for ms in mss],
            'extensions': [TypeScaleExtension().to_json()]
        }
        self.assertEqual(storage.to_json(), json_ret)
