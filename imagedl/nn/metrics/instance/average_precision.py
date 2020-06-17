import torch

from .aggregated import InstanceMetricAggregated
from .match_info import ImageEvalResults


class AveragePrecision(InstanceMetricAggregated):
    def compute_one(self, res: ImageEvalResults):
        selected = self.selection_pred(res)
        device = res.pred_area.device
        ret_val = torch.zeros(self.n_classes, device=device)
        for c in range(self.n_classes):
            indices = torch.where(res.pred_class == c)[0]
            if len(indices) == 0:
                ret_val[c] = float('nan')
                continue
            indices = indices[torch.argsort(res.conf[indices], descending=True)]

            correct = torch.ones(len(indices), device=device)
            correct *= selected[indices]
            if len(res.target_class == 0):
                correct *= res.target_class[res.pred_to_target[indices]] == c

            prec_range = torch.arange(1, len(indices) + 1, device=device)
            tp = correct.cumsum(0)
            prec = tp / prec_range
            rec = tp / (res.target_class == c).sum()

            i = len(prec_range) - 2
            while i >= 0:
                if prec[i + 1] > prec[i]:
                    prec[i] = prec[i + 1]
                i = i - 1

            rec = torch.cat([torch.tensor([0.0], device=device), rec])
            ret_val[c] = (prec * (rec[1:] - rec[:-1])).sum()
        return ret_val
