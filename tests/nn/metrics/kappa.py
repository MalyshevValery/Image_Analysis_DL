"""Test shapes and values of kappa metric"""
import unittest

import torch

from imagedl.nn.metrics import QuadraticKappa


class QuadraticKappaTestCase(unittest.TestCase):
    def test_binary(self) -> None:
        self.__compare(QuadraticKappa(),
                       torch.Tensor([1, 1, 1, 1, -1, -1]),
                       torch.Tensor([1, 1, 0, 0, 0, 1]),
                       torch.tensor([0.0]))

    def test_multi_class(self) -> None:
        self.__compare(QuadraticKappa(3, multi_label=True),
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
                       torch.Tensor([0.25, 1.0, 0.5714]))

    def test_class(self) -> None:
        self.__compare(QuadraticKappa(3),
                       torch.Tensor([[1, 0, 0], [1, 0, 0],
                                     [0, 1, 0], [0, 1, 0],
                                     [0, 0, 1], [0, 0, 1]]),
                       torch.Tensor([0, 0, 0, 1, 0, 0]),
                       torch.Tensor([0.0]))

    def test_zeros(self) -> None:
        kappa = QuadraticKappa()
        self.__compare(kappa, torch.Tensor([1, 1, 1, -1, -1, -1]),
                       torch.Tensor([0, 0, 0, 1, 1, 1]), torch.tensor(-1.0))
        self.__compare(kappa, torch.Tensor([-1, -1, -1, -1, -1, -1]),
                       torch.Tensor([0, 0, 0, 1, 1, 1]), torch.tensor(0.0))

    def test_two_updates(self) -> None:
        kappa = QuadraticKappa()
        self.__compare(kappa, torch.Tensor([-1, 1, 1, 1]),
                       torch.Tensor([0, 0, 1, 1]),
                       torch.Tensor([0.5]), False)
        self.__compare(kappa, torch.Tensor([-1, -1, 1, 1]),
                       torch.Tensor([0, 0, 0, 1]),
                       torch.Tensor([0.5294]), False)

    def __compare(self, kappa: QuadraticKappa, inp: torch.Tensor,
                  out: torch.Tensor, res: torch.Tensor,
                  reset: bool = True) -> None:
        kappa.update((inp, out))
        compare = torch.isclose(kappa.compute(), res, atol=1e-04)
        self.assertTrue(torch.all(compare).item())
        if reset:
            kappa.reset()
