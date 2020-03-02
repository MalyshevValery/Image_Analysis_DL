"""Tests for abstract extension"""
import unittest
from typing import Dict

import numpy as np

from imports.data import AbstractExtension


class AbstractExtensionTester(unittest.TestCase):
    def test_exception(self) -> None:
        class Implementation(AbstractExtension):
            """Test exceptions in AbstractExtension"""

            def __call__(self, data: np.ndarray) -> np.ndarray:
                return super().__call__(data)

            def to_json(self) -> Dict[str, object]:
                """Not implemented to_json"""
                return super().to_json()

        imp = Implementation()
        self.assertRaises(NotImplementedError, lambda: imp.to_json())
        self.assertRaises(NotImplementedError, lambda: imp(np.array([])))
