import torch

from .aggregated import InstanceMetricAggregated
from .match_info import ImageEvalResults


class AggregatedJaccardIndex(InstanceMetricAggregated):
    def compute_one(self, res: ImageEvalResults):
        selected = self.selection_target(res)
        if len(selected) == 0:
            if len(res.pred_area) == 0:
                return torch.ones(self.n_classes)
            else:
                return torch.zeros(self.n_classes)
        tp = selected
        tp[tp] &= res.target_class[tp] == res.pred_class[res.target_to_pred[tp]]
        fn = ~tp
        fp = torch.ones(len(res.pred_area), dtype=torch.bool)
        fp[res.target_to_pred[tp]] = 0

        inter_agg = self.sum_class_agg(res.target_class[tp], res.inter[tp])
        tp_union_agg = self.sum_class_agg(res.target_class[tp], res.union[tp])
        fn_union_agg = self.sum_class_agg(res.target_class[fn],
                                          res.target_area[fn])
        fp_union_agg = self.sum_class_agg(res.pred_class[fp], res.pred_area[fp])
        union_agg = fp_union_agg + tp_union_agg + fn_union_agg

        ret = inter_agg / union_agg
        ret[torch.isnan(ret)] = 1.0
        return ret
