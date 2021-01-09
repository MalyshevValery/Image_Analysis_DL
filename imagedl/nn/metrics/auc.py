from imagedl.nn.metrics.metric import UpgradedMetric
from imagedl.nn.metrics.roc import ROCCurve


class AUC(UpgradedMetric):
    def __init__(self, roc: ROCCurve):
        super().__init__()
        self._roc = roc

    def compute(self):
        tprs, fprs = self._roc.compute()
        auc = (tprs[:, 1:] * (fprs[:, 1:] - fprs[:, :-1])).sum(1)
        return auc
