"""Tests for MockDataset"""
from typing import Mapping
from unittest import TestCase

import numpy as np
from torch import from_numpy, Tensor


class MockDatasetTester(TestCase):
    """Tests for MockDataset"""

    def test(self) -> None:
        val1 = from_numpy(np.full((10, 10), 10.0))
        val2 = from_numpy(np.full((1, 1), 1.0))

        def transform(x: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
            """Test transform"""
            return {(k + '_t'): v / 2 for k, v in x.items()}

        dataset = MockDataset({'val1': val1, 'val2': val2}, 10,
                              transform=transform)
        self.assertEqual(len(dataset), 10)
        item = dataset[0]
        self.assertTrue(dataset[0]['val1_t'].mean() == 5.0)
        self.assertTrue(dataset[9]['val2_t'].mean() == 0.5)

    def test_error(self) -> None:
        tensor = from_numpy(np.zeros((1,)))
        self.assertRaises(ValueError, lambda: MockDataset(tensor, 0))
