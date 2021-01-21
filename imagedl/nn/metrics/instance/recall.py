import torch

from .aggregated import InstanceMetricAggregated
from .match_info import ImageEvalResults
from ..metric import sum_class_agg


class InstanceRecall(InstanceMetricAggregated):
    def compute_one(self, res: ImageEvalResults):
        device = res.pred_area.device
        selected = self.selection_target(res)
        selected[selected] &= res.target_class[selected] == res.pred_class[
            res.target_to_pred[selected]]
        tp = sum_class_agg(res.target_class[selected],
                           torch.ones(selected.sum(), device=device),
                           self.n_classes)
        fn = sum_class_agg(res.target_class[~selected],
                           torch.ones((~selected).sum(), device=device),
                           self.n_classes)

        ret = tp / (tp + fn)
        return ret
