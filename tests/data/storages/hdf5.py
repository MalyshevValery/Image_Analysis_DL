"""Tests for composed storage"""
import os
import unittest

import h5py
import numpy as np

from imports.data.extensions import TypeScaleExtension
from imports.data.storages import HDF5Storage


class HDF5StorageTester(unittest.TestCase):
    def setUp(self) -> None:
        """Create folder with test files"""
        self._filename = 'test.hdf5'
        self._filename_new = 'test_new.hdf5'
        file = h5py.File(self._filename, 'w')
        dataset = file.create_dataset('test', (5, 256, 256), np.float32)
        key_dataset = file.create_dataset('__keys', (5,), h5py.string_dtype())
        key_dataset[:4] = ['1', '2', '3', '4']
        dataset[:, :, :] = np.random.rand(5, 256, 256)

    def tearDown(self) -> None:
        """Remove files"""
        os.remove(self._filename)
        if os.path.isfile(self._filename_new):
            os.remove(self._filename_new)

    def test_read(self) -> None:
        type_scale = TypeScaleExtension(src_max=1.0, target_max=100)
        storage = HDF5Storage(self._filename, 'test', extensions=type_scale)
        self.assertEqual(len(storage), 4)

        all_data = np.array([storage[k] for k in storage.keys])
        self.assertGreater(np.max(all_data) + 5, 100)
        self.assertRaises(ValueError,
                          lambda: storage.save_single('-', np.array(1)))

    def test_write_old(self) -> None:
        storage = HDF5Storage(self._filename, 'test', writable=True)
        self.assertEqual(len(storage), 4)
        new_array = np.random.rand(256, 256)
        storage.save_single('-', new_array)
        storage.save_single('1', new_array)
        self.assertEqual(len(storage), 5)

        diff = np.mean(np.abs(new_array - storage['-']))
        self.assertAlmostEqual(float(diff), 0, 7)
        diff = np.mean(np.abs(new_array - storage['1']))
        self.assertAlmostEqual(float(diff), 0, 7)
        self.assertRaises(ValueError,
                          lambda: storage.save_single('--', new_array))

    def test_replace(self) -> None:
        storage = HDF5Storage(self._filename, 'test', writable=True,
                              replace=True, shape=(5, 256, 256),
                              dtype=np.float32)
        self.assertEqual(len(storage), 0)
        arr = np.random.rand(256, 256)
        storage.save_single('1', arr)
        diff = np.mean(np.abs(storage['1'] - arr))
        self.assertEqual(len(storage), 1)
        self.assertAlmostEqual(float(diff), 0, 7)

    def test_exceptions(self) -> None:
        self.assertRaises(ValueError,
                          lambda: HDF5Storage(self._filename_new, '__keys'))
        self.assertRaises(ValueError,
                          lambda: HDF5Storage(self._filename_new, 'data',
                                              writable=True))
        file = h5py.File(self._filename_new, 'w')
        file.create_dataset('temp', shape=(10,), dtype=float)
        file.close()
        self.assertRaises(ValueError,
                          lambda: HDF5Storage(self._filename_new, 'temp'))

    def test_write_new(self) -> None:
        storage = HDF5Storage(self._filename_new, 'test', writable=True,
                              shape=(5, 256, 256), dtype=np.float32)
        self.assertEqual(len(storage), 0)
        new_array = np.random.rand(5, 256, 256)
        storage.save_array(['1', '2', '3', '4', '5'], new_array)
        self.assertEqual(len(storage), 5)
        self.assertEqual({'1', '2', '3', '4', '5'}, storage.keys)
        for i in range(5):
            diff = np.mean(np.abs(new_array[i] - storage[f'{i + 1}']))
            self.assertAlmostEqual(float(diff), 0, 7)
        self.assertRaises(ValueError,
                          lambda: storage.save_single('--', new_array))

    def test_json(self) -> None:
        type_scale = TypeScaleExtension(src_max=255, target_type=np.float32,
                                        target_max=1.0)
        storage = HDF5Storage(self._filename, 'test', extensions=type_scale)
        json_ret = {
            'type': 'hdf5',
            'filename': self._filename,
            'dataset': 'test',
            'extensions': [type_scale.to_json()]
        }
        self.assertEqual(storage.to_json(), json_ret)
