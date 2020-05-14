"""Intersection over Union Scores"""
from ignite.metrics import Metric

from .confusion_matrix import ConfusionMatrix


def IoU(cm: ConfusionMatrix) -> Metric:
    """Intersection over Union score"""
    iou = cm.diag() / (cm.sum(dim=0) + cm.sum(dim=1) - cm.diag() + 1e-7)
    if cm.n_classes == 1:
        return iou[1]
    return iou


def mIoU(cm: ConfusionMatrix) -> Metric:
    """Mean intersection over Union"""
    return IoU(cm).mean()
