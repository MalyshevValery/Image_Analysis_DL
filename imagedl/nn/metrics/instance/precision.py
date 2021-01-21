import torch

from .aggregated import InstanceMetricAggregated
from .match_info import ImageEvalResults
from ..metric import sum_class_agg


class InstancePrecision(InstanceMetricAggregated):
    def compute_one(self, res: ImageEvalResults):
        selected = self.selection_target(res)
        device = res.pred_area.device
        selected[selected] &= res.target_class[selected] == res.pred_class[
            res.target_to_pred[selected]]
        tp = sum_class_agg(res.target_class[selected],
                           torch.ones(selected.sum(), device=device),
                           self.n_classes)

        fp_pred = torch.ones(len(res.pred_class), dtype=torch.bool,
                             device=device)
        fp_pred[res.target_to_pred[selected]] = 0
        fp_idx = torch.where(fp_pred)[0]
        fp = sum_class_agg(res.pred_class[fp_idx],
                           torch.ones(fp_pred.sum(), device=device),
                           self.n_classes)

        ret = tp / (tp + fp)
        return ret
