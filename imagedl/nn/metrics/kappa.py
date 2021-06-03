"""Quadratic Kappa"""
import torch

from imagedl.utils.types import MetricTransform
from .confusion_matrix import ConfusionMatrix


class QuadraticKappa(ConfusionMatrix):
    """Kappa with quadratic weights"""

    def __init__(self, n_classes: int = 1, multi_label: bool = False,
                 output_transform: MetricTransform = lambda x: x):
        super().__init__(n_classes, multi_label, output_transform)
        n = n_classes + self._is_binary if not multi_label else 2
        r = torch.arange(0, n, dtype=torch.float)
        self._weight_matrix = torch.stack([r] * n)
        self._weight_matrix -= torch.stack([r] * n, 1)
        self._weight_matrix = self._weight_matrix ** 2
        self._weight_matrix /= (n - 1) ** 2

    def compute(self) -> torch.Tensor:
        """Metric aggregation"""
        assert self._matrix.sum() > 0
        self._weight_matrix = self._weight_matrix.to(self._matrix)
        if not self._multi_label:
            matrix = self._matrix[None]
        else:
            matrix = torch.stack([self._matrix[:, i:i + 2] for i in
                                  range(0, 2 * self._n_classes, 2)])
        random = matrix.sum(2)[..., None] @ matrix.sum(1)[:, None]
        random /= matrix.sum(dim=(1, 2), keepdim=True) ** 2 + 1e-5
        matrix = matrix / (matrix.sum(dim=(1, 2), keepdim=True) + 1e-5)
        hst = (matrix * self._weight_matrix).sum(dim=(1, 2))
        rnd = (random * self._weight_matrix).sum(dim=(1, 2))
        return torch.tensor(1 - hst / (rnd + 1e-7))
