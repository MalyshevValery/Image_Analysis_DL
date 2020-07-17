import torch

from .aggregated import InstanceMetricAggregated
from .match_info import ImageEvalResults


class InstanceConfusionMatrix(InstanceMetricAggregated):
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
                matrix[p, t] += 1

        matrix[:-1, -1] = self.sum_class_agg(res.pred_class[fp],
                                             torch.ones(fp.sum(),
                                                        device=device))
        matrix[-1, :-1] = self.sum_class_agg(res.target_class[fn],
                                             torch.ones(fn.sum(),
                                                        device=device))
        tg_un, tg_cnt = res.target_class.unique(return_counts=True)
        matrix[:, tg_un] /= tg_cnt
        matrix[:, -1] /= matrix[:, -1].sum() + 1e-4
        return matrix
