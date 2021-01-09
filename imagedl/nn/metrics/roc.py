"""Confusion matrix metric"""
from typing import Tuple, List

import matplotlib.pyplot as plt
import torch
from ignite.metrics.metric import sync_all_reduce

from imagedl.data.datasets.abstract import Transform
from imagedl.nn.metrics.metric import UpgradedMetric


class ROCCurve(UpgradedMetric):
    """
    ROC curve for simple classification

    :param n_classes: Number of classes (1 for binary case)
    :param output_transform: Transform for ignite.engine output before applying
        this metric
    """

    def __init__(self, n_classes: int = 1,
                 output_transform: Transform = lambda x: x):
        super().__init__(output_transform, True)
        self._n_classes = n_classes
        self._values = []
        self._targets = []

    def _reset(self) -> None:
        self._values = []
        self._targets = []

    def _update(self, output: Tuple[torch.Tensor, torch.Tensor]) -> None:
        """Updates the metric"""
        logits, targets = output
        if self._is_binary == 1:
            probs = torch.sigmoid(logits.detach())
        else:
            probs = torch.softmax(logits.detach(), 1)
        self._values.append(probs)
        self._targets.append(targets)

    @sync_all_reduce('_updates')
    def compute(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Metric aggregation"""
        assert self._updates > 0
        values = torch.cat(self._values, 0)
        targets = torch.cat(self._targets, 0)
        tprs = []
        fprs = []
        for cls in range(0, self._n_classes):
            tar: torch.Tensor = 1.0 * (targets == cls)
            dat = values[:, cls]
            _sorted = torch.zeros(dat.shape[0] + 1)
            _sorted[1:] = tar[torch.argsort(-dat)]
            tprs.append(torch.cumsum(_sorted, 0) / tar.sum())
            _sorted[1:] = 1 - _sorted[1:]
            fprs.append(torch.cumsum(_sorted, 0) / (len(tar) - tar.sum()))
        return torch.stack(tprs), torch.stack(fprs)

    def visualize(self, value: Tuple[torch.Tensor, torch.Tensor],
                  legend: List[str] = None) -> None:
        """Visualizing of ROC curve

        :param value: Tensor value of metric to plot
        :param legend: Optional legend from config
        """
        tprs, fprs = value
        auc = (tprs[:, 1:] * (fprs[:, 1:] - fprs[:, :-1])).sum(1)
        plt.plot([0, 1], [0, 1], linestyle='--')
        for i in range(self._n_classes):
            plt.plot(fprs[i], tprs[i], label=f'{auc[i]:.3f} - {legend[i]}')
        plt.title(f'{(auc.sum() / self._n_classes):.4f}')
        plt.legend()
