import torch

from .aggregated import InstanceMetricAggregated
from .match_info import ImageEvalResults


class PanopticQuality(InstanceMetricAggregated):
    def compute_one(self, res: ImageEvalResults):
        selected = self.selection_target(res)
        if len(selected) == 0:
            if len(res.pred_area) == 0:
                return torch.ones(self.n_classes)
            else:
                return torch.zeros(self.n_classes)
        device = res.target_class.device
        selected[selected] &= res.target_class[selected] == res.pred_class[
            res.target_to_pred[selected]]
        tp = self.sum_class_agg(res.target_class[selected],
                                torch.ones(selected.sum(), device=device))
        fn = self.sum_class_agg(res.target_class[~selected],
                                torch.ones((~selected).sum(), device=device))

        fp_pred = torch.ones(len(res.pred_class), dtype=torch.bool,
                             device=device)
        fp_pred[res.target_to_pred[selected]] = 0
        fp_idx = torch.where(fp_pred)[0]
        fp = self.sum_class_agg(res.pred_class[fp_idx],
                                torch.ones(fp_pred.sum(), device=device))

        ious = self.sum_class_agg(res.target_class[selected],
                                  res.ious[selected])
        ret = ious / (tp + 0.5 * fp + 0.5 * fn)
        ret[torch.isnan(ret)] = 1.0
        return ret
