class MeanMetric(ignite.metrics.Metric):
    def __init__(self, source):
        self._source = source
        super().__init__()

    def reset(self):
        pass

    def update(self, output):
        pass

    def compute(self):
        return self._source.compute().mean().item()


# FINISHED
class InstanceMetricAggregated(ignite.metrics.Metric, metaclass=abc.ABCMeta):
    def __init__(self, imi: InstanceMatchInfo, iou_thresh=0.0, conf_thresh=0.0):
        self._imi = imi
        self._iou_thresh = iou_thresh
        self._conf_thresh = conf_thresh
        super().__init__()

    @property
    def n_classes(self):
        return self._imi.n_classes

    def compute(self):
        res = []
        for r in self._imi.compute():
            res.append(self.compute_one(r))
        res = torch.stack(res, dim=0)
        return res.mean(dim=0)

    @abc.abstractmethod
    def compute_one(self, res: ImageEvalResults):
        raise NotImplementedError()

    def sum_class_agg(self, labels, values):
        res = torch.zeros(self.n_classes, device=values.device)
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

    def reset(self):
        pass

    def update(self, output):
        pass

    def mean(self):
        return MeanMetric(self)
