"""Test shapes and values of recall metric"""
import unittest

import torch

from imagedl.nn.metrics import Recall


class RecallTestCase(unittest.TestCase):
    def test_binary(self) -> None:
        self.__compare(Recall(),
                       torch.Tensor([1, 1, 1, 1, -1, -1]),
                       torch.Tensor([1, 1, 0, 0, 0, 1]),
                       torch.Tensor([0.6667]))

    def test_multi_class(self) -> None:
        self.__compare(Recall(3, multi_label=True),
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
                       torch.Tensor([0.75, 1.0, 1.0]))

    def test_class(self) -> None:
        self.__compare(Recall(3),
                       torch.Tensor([[1, 0, 0], [1, 0, 0],
                                     [0, 1, 0], [0, 1, 0],
                                     [0, 0, 1], [0, 0, 1]]),
                       torch.Tensor([0, 0, 0, 1, 0, 0]),
                       torch.Tensor([0.4, 1.0, 0.0]))

    def test_zeros(self) -> None:
        rec = Recall()
        zer = torch.zeros(1)
        self.__compare(rec, torch.Tensor([1, 1, 1, -1, -1, -1]),
                       torch.Tensor([0, 0, 0, 1, 1, 1]), zer)
        self.__compare(rec, torch.Tensor([-1, -1, -1, 1, 1, 1]),
                       torch.Tensor([0, 0, 0, 0, 0, 0]), zer)

    def test_two_updates(self) -> None:
        rec = Recall()
        self.__compare(rec, torch.Tensor([-1, -1, 1, 1]),
                       torch.Tensor([0, 1, 1, 1]),
                       torch.Tensor([0.6667]), False)
        self.__compare(rec, torch.Tensor([-1, -1, -1, 1]),
                       torch.Tensor([0, 0, 1, 1]),
                       torch.Tensor([0.6]), False)

    def __compare(self, rec: Recall, inp: torch.Tensor, out: torch.Tensor,
                  res: torch.Tensor, reset: bool = True) -> None:
        rec.update((inp, out))
        compare = torch.isclose(rec.compute(), res, 1e-04)
        self.assertTrue(torch.all(compare).item())
        if reset:
            rec.reset()
