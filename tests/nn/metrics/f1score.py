"""F1 Score metric"""
import unittest

import torch

from imagedl.nn.metrics import Precision, Recall
from imagedl.nn.metrics.f1score import F1Score


class F1TestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        """Setup values to be used later"""
        cls.prec = Precision()
        cls.rec = Recall()
        cls.f1 = F1Score(cls.prec, cls.rec)

    def test_binary(self) -> None:
        self.__compare(1, False,
                       torch.Tensor([1, 1, 1, 1, -1, -1]),
                       torch.Tensor([1, 1, 0, 0, 0, 1]),
                       torch.tensor([0.5714]))

    def test_multi_class(self) -> None:
        self.__compare(3, True,
                       torch.Tensor([
                           [1, 1, 1], [1, 1, 1],
                           [1, 1, 1], [1, -1, 1],
                           [-1, -1, 1], [-1, -1, -1]
                       ]),
                       torch.Tensor([
                           [1, 1, 1], [1, 1, 1],
                           [0, 1, 1], [1, 0, 1],
                           [1, 0, 0], [0, 0, 0]
                       ]),
                       torch.Tensor([0.75, 1.0, 0.8889]))

    def test_class(self) -> None:
        self.__compare(3, False,
                       torch.Tensor([[1, 0, 0], [1, 0, 0],
                                     [0, 1, 0], [0, 1, 0],
                                     [0, 0, 1], [0, 0, 1]]),
                       torch.Tensor([0, 0, 0, 1, 0, 0]),
                       torch.Tensor([0.5714, 0.6667, 0.0]))

    def test_zeros(self) -> None:
        zer = torch.tensor([0.0])
        self.__compare(1, False, torch.Tensor([1, 1, 1, -1, -1, -1]),
                       torch.Tensor([0, 0, 0, 1, 1, 1]), zer)
        self.__compare(1, False, torch.Tensor([-1, -1, -1, -1, -1, -1]),
                       torch.Tensor([0, 0, 0, 1, 1, 1]), zer)

    def test_two_updates(self) -> None:
        self.__compare(1, False, torch.Tensor([-1, 1, 1, 1]),
                       torch.Tensor([0, 0, 1, 1]),
                       torch.Tensor([0.8]))
        self.__compare(1, False, torch.Tensor([-1, -1, 1, 1]),
                       torch.Tensor([0, 0, 0, 1]),
                       torch.Tensor([0.75]), True)

    def __compare(self, n_classes: int, multi_label: bool, inp: torch.Tensor,
                  out: torch.Tensor, res: torch.Tensor,
                  keep: bool = False) -> None:
        if not keep:
            self.prec = Precision(n_classes, multi_label=multi_label)
            self.rec = Recall(n_classes, multi_label=multi_label)
            self.f1 = F1Score(self.prec, self.rec)
        self.prec.update((inp, out))
        self.rec.update((inp, out))
        compare = torch.isclose(self.f1.compute(), res, 1e-04)
        self.assertTrue(torch.all(compare).item())
