from typing import Tuple
import ignite
import torch

from .aji import AggregatedJaccardIndex


class APFromAji(ignite.metrics.Metric):
    """Average precision that is calculated using AJI metric results"""
    def __init__(self, aji: AggregatedJaccardIndex, iou_thresh: float):
        self._aji = aji
        self._iou_thresh = iou_thresh
        super().__init__()

    def reset(self) -> None:
        """Reset the metric"""
        pass

    def update(self, output: Tuple[torch.Tensor, torch.Tensor]) -> None:
        """Updates the metric"""
        pass

    def compute(self) -> float:
        """Metric aggregation"""
        score = 0.0
        c = 0
        for eval_res in self._aji.image_results:
            idx = eval_res.ious > self._iou_thresh
            tp = len(eval_res.prediction_indexes[idx].unique())
            fp = eval_res.n_predictions - tp
            score += tp / (tp + fp)
            c += 1
        return score / c
