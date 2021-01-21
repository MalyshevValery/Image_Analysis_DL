from typing import List

import torch
from seaborn import heatmap

from .aggregated import InstanceMetricAggregated
from .match_info import ImageEvalResults, InstanceMatchInfo
from ..metric import sum_class_agg


class InstanceConfusionMatrix(InstanceMetricAggregated):
    def __init__(self, imi: InstanceMatchInfo, iou_thresh=0.0,
                 conf_thresh=0.0):
        super().__init__(imi, iou_thresh, conf_thresh, True)

    def compute_one(self, res: ImageEvalResults):
        selected = self.selection_target(res)
        fn = ~selected
        fp = torch.ones(len(res.pred_area), dtype=torch.bool)
        device = res.target_area.device
        matrix = torch.zeros(self.n_classes + 1, self.n_classes + 1,
                             device=device)

        if len(selected) > 0:
            fp[res.target_to_pred[selected]] = 0

            classes = zip(res.pred_class[res.target_to_pred[selected]],
                          res.target_class[selected])
            for p, t in classes:
                matrix[p + 1, t + 1] += 1

        matrix[1:, 0] = sum_class_agg(res.pred_class[fp],
                                      torch.ones(fp.sum(), device=device),
                                      self.n_classes)
        matrix[0, 1:] = sum_class_agg(res.target_class[fn],
                                      torch.ones(fn.sum(), device=device),
                                      self.n_classes)
        return matrix / (matrix.sum() + 1e-7)

    def visualize(self, value: torch.Tensor, legend: List[str] = None) -> None:
        """Visualizing method for Confusion Matrix

        :param value: Tensor value of metric to plot
        :param legend: Optional legend from config
        """
        heatmap(value.cpu().numpy(), annot=True, xticklabels=legend,
                yticklabels=legend)
