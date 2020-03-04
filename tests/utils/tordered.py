"""Tests for TOrderedSet"""
import pickle
import unittest
from typing import Iterator

from imports.utils import TOrderedSet


class TOrderedSetTester(unittest.TestCase):
    def test_all_methods(self) -> None:
        test_set = TOrderedSet((1, 2, 3))
        self.assertEqual(len(test_set), 3)
        self.assertSequenceEqual(test_set[0:3], [1, 2, 3])
        self.assertEqual(test_set.copy(), test_set)
        self.assertNotEqual(id(test_set), id(test_set.copy()))
        self.assertEqual(test_set, pickle.loads(pickle.dumps(test_set)))
        self.assertTrue(1 in test_set)
        self.assertFalse(0 in test_set)
        self.assertTrue(test_set == [1, 2, 3])
        test_set.add(4)
        self.assertEqual(len(test_set), 4)
        self.assertEqual(test_set.pop(), 4)
        self.assertEqual(len(test_set), 3)
        self.assertEqual(test_set.index(2), 1)
        self.assertSequenceEqual([i for i in test_set], [1, 2, 3])
        iterator: Iterator[int] = reversed(test_set)
        self.assertSequenceEqual([i for i in iterator],
                                 [3, 2, 1])
        test_set.update((4, 5))
        self.assertEqual(len(test_set), 5)
        test_set.discard(4)
        self.assertEqual(test_set.index(5), 3)
        test_set.discard(5)
        self.assertEqual(repr(test_set), 'TOrderedSet([1, 2, 3])')

        test_set2 = TOrderedSet((3, 4, 5))
        self.assertSequenceEqual(test_set.union(test_set2), [1, 2, 3, 4, 5])
        self.assertSetEqual(set(test_set & test_set2), {3})
        self.assertSequenceEqual(test_set.intersection(test_set2), [3])
        self.assertSequenceEqual(test_set.difference(test_set2), [1, 2])
        self.assertSequenceEqual(test_set.symmetric_difference(test_set2),
                                 [1, 2, 4, 5])
        copy = test_set.copy()
        copy.difference_update(test_set2)
        self.assertSequenceEqual(copy, [1, 2])

        copy = test_set.copy()
        copy.symmetric_difference_update(test_set2)
        self.assertSequenceEqual(copy, [1, 2, 4, 5])

        copy = test_set.copy()
        copy.intersection_update(test_set2)
        self.assertSequenceEqual(copy, [3])

        self.assertTrue(test_set.issubset([1, 2, 3, 4]))
        self.assertFalse(test_set.issubset([1, 2, 4]))

        self.assertTrue(test_set.issuperset([1, 2]))
        self.assertFalse(test_set.issuperset([2, 3, 4]))

        test_set.clear()
        self.assertEqual(len(test_set), 0)
