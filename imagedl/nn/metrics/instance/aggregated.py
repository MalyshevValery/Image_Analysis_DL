import abc

import torch
from ignite.metrics.metric import Metric, reinit__is_reduced

from .match_info import InstanceMatchInfo, ImageEvalResults


class MeanMetric(Metric):
    def __init__(self, source):
        self._source = source
        super().__init__()

    @reinit__is_reduced
    def reset(self):
        pass

    @reinit__is_reduced
    def update(self, output):
        pass

    def compute(self):
        return self._source.compute().mean().item()


class InstanceMetricAggregated(Metric, metaclass=abc.ABCMeta):
    def __init__(self, imi: InstanceMatchInfo, iou_thresh=0.0, conf_thresh=0.0):
        self._imi = imi
        self._iou_thresh = iou_thresh
        self._conf_thresh = conf_thresh
        super().__init__()

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

    def sum_class_agg(self, labels, values):
        res = torch.zeros(self.n_classes if self.n_classes is not None else 1, device=values.device)
        res.scatter_add_(0, labels, values)
        return res

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

    @reinit__is_reduced
    def reset(self):
        pass

    @reinit__is_reduced
    def update(self, output):
        pass

    def mean(self):
        return MeanMetric(self)
