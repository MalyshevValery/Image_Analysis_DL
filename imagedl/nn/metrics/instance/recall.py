import torch

from .aggregated import InstanceMetricAggregated
from .match_info import ImageEvalResults


class InstanceRecall(InstanceMetricAggregated):
    def compute_one(self, res: ImageEvalResults):
        device = res.pred_area.device
        selected = self.selection_target(res)
        if len(selected) == 0:
            return torch.ones(self.n_classes)
        selected[selected] &= res.target_class[selected] == res.pred_class[
            res.target_to_pred[selected]]
        tp = self.sum_class_agg(res.target_class[selected],
                                torch.ones(selected.sum(), device=device))
        fn = self.sum_class_agg(res.target_class[~selected],
                                torch.ones((~selected).sum(), device=device))

        ret = tp / (tp + fn)
        return ret
