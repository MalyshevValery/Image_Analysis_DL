"""F1 score"""
import torch

from .metric import UpgradedMetric
from .precision import Precision
from .recall import Recall


class F1Score(UpgradedMetric):
    """F1 score which is calculated form precision and recall"""

    def __init__(self, prec: Precision, rec: Recall):
        self._prec = prec
        self._rec = rec
        super().__init__()

    def compute(self) -> torch.Tensor:
        """Computes F1 score with precision and recall"""
        pr = self._prec.compute()
        rec = self._rec.compute()
        return 2 * (pr * rec) / (pr + rec + 1e-7)
