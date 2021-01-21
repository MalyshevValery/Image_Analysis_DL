import torch

from .aggregated import InstanceMetricAggregated
from .match_info import ImageEvalResults
from ..metric import sum_class_agg


class AggregatedJaccardIndex(InstanceMetricAggregated):
    def compute_one(self, res: ImageEvalResults):
        selected = self.selection_target(res)
        tp = selected
        tp[tp] &= res.target_class[tp] == res.pred_class[res.target_to_pred[tp]]
        fn = ~tp
        fp = torch.ones(len(res.pred_area), dtype=torch.bool)
        fp[res.target_to_pred[tp]] = 0

        inter_agg = sum_class_agg(res.target_class[tp], res.inter[tp],
                                  self.n_classes)
        tp_union_agg = sum_class_agg(res.target_class[tp], res.union[tp],
                                     self.n_classes)
        fn_union_agg = sum_class_agg(res.target_class[fn], res.target_area[fn],
                                     self.n_classes)
        fp_union_agg = sum_class_agg(res.pred_class[fp], res.pred_area[fp],
                                     self.n_classes)
        union_agg = fp_union_agg + tp_union_agg + fn_union_agg

        ret = inter_agg / union_agg
        return ret
