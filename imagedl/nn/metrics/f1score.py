from ignite.metrics import Metric
from ignite.metrics.metric import reinit__is_reduced

from imagedl.nn.metrics.instance import MeanMetric


class F1Score(Metric):
    def __init__(self, prec, rec):
        self._prec = prec
        self._rec = rec
        super().__init__()

    def compute(self):
        pr = self._prec.compute()
        rec = self._rec.compute()
        return 2 * (pr * rec) / (pr + rec + 1e-7)

    @reinit__is_reduced
    def reset(self):
        pass

    def update(self, output):
        pass

    def mean(self):
        return MeanMetric(self)
