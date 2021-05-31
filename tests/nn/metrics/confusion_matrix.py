"""Confusion matrix metric"""
import unittest

import torch

from imagedl.nn.metrics.confusion_matrix import ConfusionMatrix


class ConfusionTestCase(unittest.TestCase):
    def test_binary(self) -> None:
        self.__compare(ConfusionMatrix(),
                       torch.Tensor([1, 1, 1, -1]),
                       torch.Tensor([1, 1, 0, 0]),
                       torch.tensor([
                           [.5, 0.],
                           [.5, 1.]
                       ]))

    def test_multi_class(self) -> None:
        self.__compare(ConfusionMatrix(3, multi_label=True),
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
                       torch.Tensor([
                           [.5, .25, 1., 0., .5, 0.],
                           [.5, .75, 0., 1., .5, 1.]
                       ]))

    def test_class(self) -> None:
        self.__compare(ConfusionMatrix(3),
                       torch.Tensor([[1, 0, 0], [1, 0, 0],
                                     [0, 1, 0], [0, 1, 0],
                                     [0, 0, 1], [0, 0, 1]]),
                       torch.Tensor([0, 0, 0, 1, 0, 0]),
                       torch.Tensor([
                           [0.4, 0.0, 0.0],
                           [0.2, 1.0, 0.0],
                           [0.4, 0.0, 0.0]
                       ]))

    def test_zeros(self) -> None:
        cm = ConfusionMatrix()
        zer = torch.zeros(2, 2)
        zer[1, 0] = 1.0
        self.__compare(cm, torch.Tensor([1, 1, 1]),
                       torch.Tensor([0, 0, 0]), zer)
        zer = zer.T
        self.__compare(cm, torch.Tensor([0, 0, 0]),
                       torch.Tensor([1, 1, 1]), zer)

    def test_two_updates(self) -> None:
        cm = ConfusionMatrix()
        self.__compare(cm, torch.Tensor([-1, 1, 1, 1]),
                       torch.Tensor([0, 0, 1, 1]),
                       torch.Tensor([
                           [0.5, 0.0],
                           [0.5, 1.0]
                       ]), False)
        self.__compare(cm, torch.Tensor([-1, -1, 1, 1]),
                       torch.Tensor([0, 0, 0, 1]),
                       torch.Tensor([
                           [0.6, 0.0],
                           [0.4, 1.0]
                       ]), False)

    def __compare(self, cm: ConfusionMatrix, inp: torch.Tensor,
                  out: torch.Tensor, res: torch.Tensor,
                  reset: bool = True) -> None:
        cm.update((inp, out))
        compare = torch.isclose(cm.compute(), res, 1e-04)
        self.assertTrue(torch.all(compare).item())
        if reset:
            cm.reset()
