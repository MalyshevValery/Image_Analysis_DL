"""AUC"""
import torch

from imagedl.nn.metrics.metric import UpgradedMetric
from imagedl.nn.metrics.roc import ROCCurve


class AUC(UpgradedMetric):
    """Area under ROC curve"""

    def __init__(self, roc: ROCCurve):
        super().__init__()
        self._roc = roc

    def compute(self) -> torch.Tensor:
        """Computes AUC from ROCCurve data"""
        values = self._roc.compute()
        tprs, fprs = values[0], values[1]
        auc = (tprs[:, 1:] * (fprs[:, 1:] - fprs[:, :-1])).sum(1)
        return auc
