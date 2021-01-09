"""Intersection over Union Scores"""
from ignite.metrics import Metric

from .confusion_matrix import ConfusionMatrix


def iou(cm: ConfusionMatrix) -> Metric:
    """Intersection over Union score"""
    iou_val = cm.diag() / (cm.sum(dim=0) + cm.sum(dim=1) - cm.diag() + 1e-7)
    if cm.n_classes == 1:
        return iou_val[1]
    return iou_val


def mean_iou(cm: ConfusionMatrix) -> Metric:
    """Mean intersection over Union"""
    return iou(cm).mean()
