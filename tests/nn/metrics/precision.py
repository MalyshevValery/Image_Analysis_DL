"""Test shapes and values of precision metric"""
import unittest

import torch

from imagedl.nn.metrics import Precision


class PrecisionTestCase(unittest.TestCase):
    def test_binary(self) -> None:
        self.__compare(Precision(),
                       torch.Tensor([1, 1, 1, 1, -1, -1]),
                       torch.Tensor([1, 1, 0, 0, 0, 1]),
                       torch.tensor([0.5]))

    def test_multi_class(self) -> None:
        self.__compare(Precision(3, multi_label=True),
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
                       torch.Tensor([0.75, 1.0, 0.8]))

    def test_class(self) -> None:
        self.__compare(Precision(3),
                       torch.Tensor([[1, 0, 0], [1, 0, 0],
                                     [0, 1, 0], [0, 1, 0],
                                     [0, 0, 1], [0, 0, 1]]),
                       torch.Tensor([0, 0, 0, 1, 0, 0]),
                       torch.Tensor([1.0, 0.5, 0.0]))

    def test_zeros(self) -> None:
        prec = Precision()
        zer = torch.zeros(1)
        self.__compare(prec, torch.Tensor([1, 1, 1, -1, -1, -1]),
                       torch.Tensor([0, 0, 0, 1, 1, 1]), zer)
        self.__compare(prec, torch.Tensor([-1, -1, -1, -1, -1, -1]),
                       torch.Tensor([0, 0, 0, 1, 1, 1]), zer)

    def test_two_updates(self) -> None:
        prec = Precision()
        self.__compare(prec, torch.Tensor([-1, 1, 1, 1]),
                       torch.Tensor([0, 0, 1, 1]),
                       torch.Tensor([0.6667]), False)
        self.__compare(prec, torch.Tensor([-1, -1, 1, 1]),
                       torch.Tensor([0, 0, 0, 1]),
                       torch.Tensor([0.6]), False)

    def __compare(self, prec: Precision, inp: torch.Tensor, out: torch.Tensor,
                  res: torch.Tensor, reset: bool = True) -> None:
        prec.update((inp, out))
        compare = torch.isclose(prec.compute(), res, 1e-04)
        self.assertTrue(torch.all(compare).item())
        if reset:
            prec.reset()
