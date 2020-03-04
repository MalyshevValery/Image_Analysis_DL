"""Tests for AbstractStorage"""
import unittest
from typing import Dict

import numpy as np

from imports.data.storages import AbstractStorage


class AbstractStorageTester(unittest.TestCase):
    class __DummyStorage(AbstractStorage):

        def __getitem__(self, item: str) -> np.ndarray:
            return np.array(float(item))

        def save_single(self, key: str, data: np.ndarray) -> None:
            """Dummy saves single"""
            self._add_keys(key)

        def to_json(self) -> Dict[str, object]:
            """Dummy to json"""
            return super().to_json()

    def test_properties(self) -> None:
        ds = AbstractStorageTester.__DummyStorage({'0', '1'}, writable=True)
        self.assertEqual(ds.keys, {'0', '1'})
        self.assertTrue(ds.writable)

    def test_save_array(self) -> None:
        ds = AbstractStorageTester.__DummyStorage({'0'}, writable=True)
        ds.save_array(['1', '2'], np.array([0, 0]))
        self.assertEqual(ds['0'], 0.0)
        self.assertEqual(ds['1'], 1.0)
        self.assertEqual(ds['2'], 2.0)

    def test_abstract_methods(self) -> None:
        ds = AbstractStorageTester.__DummyStorage({'1', '2'})
        self.assertRaises(NotImplementedError, lambda: ds.to_json())
        self.assertRaises(ValueError, lambda: ds.save_array(['1'], np.array(1)))
        ds.save_single('0', np.array(1.0))
        self.assertEqual(ds['0'], 0.0)
        self.assertEqual(ds['1'], 1.0)
        self.assertEqual(ds['2'], 2.0)

    def test_exception(self) -> None:
        class __EmptyStorage(AbstractStorage):
            def __getitem__(self, item: str) -> np.ndarray:
                return super().__getitem__(item)

            def save_single(self, key: str, data: np.ndarray) -> None:
                """Dummy saves single"""
                return super().save_single(key, data)

            def to_json(self) -> Dict[str, object]:
                """Dummy to json"""
                return super().to_json()

        storage = __EmptyStorage(set())
        self.assertRaises(NotImplementedError, lambda: storage['-'])
        self.assertRaises(NotImplementedError,
                          lambda: storage.save_single('1', np.array([])))
