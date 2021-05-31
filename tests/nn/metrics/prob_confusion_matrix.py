"""Probability Confusion matrix metric"""
import unittest

import torch

from imagedl.nn.metrics.prob_confusion_matrix import ProbConfusionMatrix


class ProbConfusionTestCase(unittest.TestCase):
    def test_binary(self) -> None:
        s = torch.sigmoid(torch.tensor(1.0))
        self.__compare(ProbConfusionMatrix(),
                       torch.Tensor([1, 1, 1, -1]),
                       torch.Tensor([1, 1, 0, 0]),
                       torch.tensor([
                           [.5, 1 - s],
                           [.5, s]
                       ]))

    def test_multi_class(self) -> None:
        self.__compare(ProbConfusionMatrix(3, multi_label=True),
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
                           [.5, .3845, .7311, .2689, .5, .2689],
                           [.5, .6155, .2689, .7311, .5, .7311]
                       ]))

    def test_class(self) -> None:
        s = torch.softmax(torch.tensor([1., .0, .0]), 0)
        self.__compare(ProbConfusionMatrix(3),
                       torch.Tensor([[1, 0, 0], [1, 0, 0],
                                     [0, 1, 0], [0, 1, 0],
                                     [0, 0, 1], [0, 0, 1]]),
                       torch.Tensor([0, 0, 0, 1, 0, 0]),
                       torch.Tensor([
                           [0.3576, s[1], 0.0],
                           [0.2848, s[0], 0.0],
                           [0.3576, s[2], 0.0]
                       ]))

    def test_two_updates(self) -> None:
        cm = ProbConfusionMatrix()
        self.__compare(cm, torch.Tensor([-1, 1, 1, 1]),
                       torch.Tensor([0, 0, 1, 1]),
                       torch.Tensor([
                           [0.5, .2689],
                           [0.5, .7311]
                       ]), False)
        self.__compare(cm, torch.Tensor([-1, -1, 1, 1]),
                       torch.Tensor([0, 0, 0, 1]),
                       torch.Tensor([
                           [0.5462, .2689],
                           [0.4538, .7311]
                       ]), False)

    def __compare(self, cm: ProbConfusionMatrix, inp: torch.Tensor,
                  out: torch.Tensor, res: torch.Tensor,
                  reset: bool = True) -> None:
        cm.update((inp, out))
        compare = torch.isclose(cm.compute(), res, atol=1e-04)
        self.assertTrue(torch.all(compare).item())
        if reset:
            cm.reset()
