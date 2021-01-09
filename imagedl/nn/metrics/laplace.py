"""Confusion matrix metric"""
from typing import Tuple

import numpy as np
import torch

from imagedl.data.datasets.abstract import Transform
from .metric import UpgradedMetric


class Laplace(UpgradedMetric):
    """Laplacian metric for classification"""

    def __init__(self, clips: Tuple[float, float] = None,
                 output_transform: Transform = lambda x: x):
        super().__init__(output_transform=output_transform)
        self.res = 0.0
        self.n = 0.0
        self.clips = clips

    def _reset(self) -> None:
        """Resets the metric"""
        self.res = 0.0
        self.n = 0.0

    def _update(self, output: Tuple[torch.Tensor, torch.Tensor]) -> None:
        """Updates the metric"""
        logits, targets = output
        mean, std = logits
        delta = torch.abs(mean - targets)
        if self.clips is not None:
            delta[delta > self.clips[0]] = self.clips[0]
            std[std < self.clips[1]] = self.clips[1]
        laplace = -np.sqrt(2) * delta / std - torch.log(np.sqrt(2) * std)
        self.res += laplace.sum()
        self.n += len(laplace)

    def compute(self) -> float:
        """Metric aggregation"""
        return self.res / self.n
