"""Tests for types"""
import unittest
from typing import Sequence

from imports.utils import to_seq, seq_apply


class TypesTester(unittest.TestCase):
    def test_to_seq(self) -> None:
        self.assertSequenceEqual(to_seq([1, 2, 3]), [1, 2, 3])
        self.assertSequenceEqual(to_seq((1, 2, 3)), [1, 2, 3])
        self.assertSequenceEqual(to_seq(1), [1])

    def test_seq_apply(self) -> None:
        def to_str_x2(arr: Sequence[int]) -> Sequence[str]:
            """Multiplies int by two and transforms to string"""
            return [str(x * 2) for x in arr]

        true_seq = ['2', '4', '6']
        self.assertSequenceEqual(seq_apply([1, 2, 3], to_str_x2), true_seq)
        self.assertSequenceEqual(seq_apply((1, 2, 3), to_str_x2), true_seq)
        self.assertSequenceEqual(seq_apply(1, to_str_x2), '2')
