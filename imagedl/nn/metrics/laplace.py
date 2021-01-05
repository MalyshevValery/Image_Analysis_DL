"""Confusion matrix metric"""
from typing import Tuple

import numpy as np
import torch
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce

from imagedl.data.datasets.abstract import Transform


class Laplace(Metric):
    def __init__(self, clips=None,
                 output_transform: Transform = lambda x: x):
        super().__init__(output_transform=output_transform)
        self.res = 0.0
        self.n = 0.0
        self.clips = clips

    @reinit__is_reduced
    def reset(self) -> None:
        """Resets the metric"""
        self.res = 0.0
        self.n = 0.0

    @reinit__is_reduced
    def update(self, output: Tuple[torch.Tensor, torch.Tensor]) -> None:
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

    @sync_all_reduce('_updates', '_matrix')
    def compute(self) -> torch.Tensor:
        """Metric aggregation"""
        return self.res / self.n
