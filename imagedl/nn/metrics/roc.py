"""Confusion matrix metric"""
from typing import Tuple, List

import matplotlib.pyplot as plt
import torch

from imagedl.utils.types import MetricTransform
from .metric import UpgradedMetric


def _calc_class(tar: torch.Tensor,
                val: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    _sorted = torch.zeros(val.shape[0] + 1, device=tar.device)
    _sorted[1:] = tar[torch.argsort(-val)]
    tpr = torch.cumsum(_sorted, 0) / (tar.sum() + 1e-7)
    _sorted[1:] = 1 - _sorted[1:]
    fpr = torch.cumsum(_sorted, 0) / (len(tar) - tar.sum() + 1e-7)
    return tpr, fpr


class ROCCurve(UpgradedMetric):
    """
    ROC curve for simple classification

    :param n_classes: Number of classes (1 for binary case)
    :param output_transform: Transform for ignite.engine output before applying
        this metric
    """

    def __init__(self, n_classes: int = 1,
                 output_transform: MetricTransform = lambda x: x,
                 multi_label: bool = False):
        super().__init__(output_transform, True)
        self._n_classes = n_classes
        self._multi_label = multi_label
        if n_classes == 1 and self._multi_label:
            raise ValueError('For multi-label number of classes must exceed 1')
        self._values: List[torch.Tensor] = []
        self._targets: List[torch.Tensor] = []

    def _reset(self) -> None:
        self._values = []
        self._targets = []

    def _update(self, output: Tuple[torch.Tensor, torch.Tensor]) -> None:
        """Updates the metric"""
        logits, targets = output
        if self._n_classes == 1 or self._multi_label:
            probs = torch.sigmoid(logits.detach())
        else:
            probs = torch.softmax(logits.detach(), 1)
        self._values.append(probs)
        self._targets.append(targets)

    def compute(self) -> torch.Tensor:
        """Metric aggregation"""
        assert self._updates > 0
        values = torch.cat(self._values, 0)
        targets = torch.cat(self._targets, 0)
        if self._n_classes == 1:
            tpr, fpr = _calc_class(targets, values)
            tprs = tpr[None]
            fprs = fpr[None]
        else:
            tpr_list = []
            fpr_list = []
            for cls in range(0, self._n_classes):
                if self._multi_label:
                    tar: torch.Tensor = targets[:, cls]
                else:
                    tar = 1.0 * torch.eq(targets, cls)
                if len(values.shape) == 1:
                    dat = values
                else:
                    dat = values[:, cls]
                tpr, fpr = _calc_class(tar, dat)
                tpr_list.append(tpr)
                fpr_list.append(fpr)
            tprs, fprs = torch.stack(tpr_list), torch.stack(fpr_list)
        return torch.stack([tprs, fprs])

    def visualize(self, value: torch.Tensor, legend: List[str] = None) -> None:
        """Visualizing of ROC curve

        :param value: Tensor value of metric to plot
        :param legend: Optional legend from config
        """
        tprs, fprs = value[0], value[1]
        auc = (tprs[:, 1:] * (fprs[:, 1:] - fprs[:, :-1])).sum(1)
        plt.plot([0, 1], [0, 1], linestyle='--')
        for i in range(self._n_classes):
            label = f'{auc[i]:.3f}'
            if legend is not None:
                label += f' - {legend[i]}'
            plt.plot(fprs[i].cpu(), tprs[i].cpu(), label=label)

        plt.title(f'{(auc.sum() / self._n_classes):.4f}')
        plt.legend()
