import abc

import torch

from .match_info import InstanceMatchInfo, ImageEvalResults
from ..metric import UpgradedMetric


class InstanceMetricAggregated(UpgradedMetric, metaclass=abc.ABCMeta):
    def __init__(self, imi: InstanceMatchInfo, iou_thresh=0.0,
                 conf_thresh=0.0, vis=False):
        self._imi = imi
        self._iou_thresh = iou_thresh
        self._conf_thresh = conf_thresh
        super().__init__(vis=vis)

    @property
    def n_classes(self):
        return self._imi.n_classes if self._imi.n_classes is not None else 1

    def compute(self):
        val = self.compute_one(self._imi.compute(True))
        val[torch.isnan(val)] = 0.0
        return val

    @abc.abstractmethod
    def compute_one(self, res: ImageEvalResults):
        raise NotImplementedError()

    def selection_target(self, res: ImageEvalResults):
        sel = res.target_to_pred != -1
        if len(sel) == 0:
            return sel
        sel &= res.ious >= self._iou_thresh
        if res.conf is not None:
            sel[sel] &= res.conf[res.target_to_pred[sel]] >= self._conf_thresh
        return sel

    def selection_pred(self, res: ImageEvalResults):
        sel = res.pred_to_target != -1
        if len(sel) == 0:
            return sel
        sel[sel] &= res.ious[res.pred_to_target[sel]] >= self._iou_thresh
        if res.conf is not None:
            sel &= res.conf > self._conf_thresh
        return sel
