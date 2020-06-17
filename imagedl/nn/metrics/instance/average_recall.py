import torch

from .aggregated import InstanceMetricAggregated
from .match_info import ImageEvalResults


class AverageRecall(InstanceMetricAggregated):
    def compute_one(self, res: ImageEvalResults):
        selected = self.selection_target(res)
        device = res.pred_area.device

        ret_val = torch.zeros(self.n_classes, device=device)
        for c in range(self.n_classes):
            class_selected = res.target_class == c
            if class_selected.sum() == 0:
                ret_val[c] = float('nan')
                continue
            class_selected &= selected
            class_selected[class_selected] &= res.pred_class[res.target_to_pred[class_selected]] == c

            ious = res.ious[class_selected]
            ious = ious.sort().values

            tp_range = torch.arange(class_selected.sum(), 0, -1,
                                    device=device).float()
            rec = tp_range / (res.target_class == c).sum()
            ious = torch.cat([torch.tensor([0.0], device=device), ious])
            ret_val[c] = (rec * (ious[1:] - ious[:-1])).sum()
        return ret_val
