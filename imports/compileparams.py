"""Wrapper for compile params"""
import copy

import segmentation_models as seg

from imports.jsonserializable import JSONSerializable

LOSSES = {
    'binary_focal': seg.losses.BinaryFocalLoss(),
    'categorical_focal': seg.losses.CategoricalFocalLoss(),
    'dice': seg.losses.DiceLoss,
    'jaccard': seg.losses.JaccardLoss
}

METRICS = {
    'iou': seg.metrics.IOUScore(name='iou'),
    'f1': seg.metrics.FScore(beta=1, name='f1'),
    'f2': seg.metrics.FScore(beta=2, name='f2'),
    'precision': seg.metrics.Precision(name='precision'),
    'recall': seg.metrics.Recall(name='recall')
}


class CompileParams(JSONSerializable):
    """Wraps string to metrics and loss conversion"""

    def __init__(self, json):
        self._json = copy.deepcopy(json)
        self._params = copy.deepcopy(json)
        if self._params.get('loss', None) in LOSSES:
            self._params['loss'] = LOSSES[self._params['loss']]()
        if 'metrics' in self._params:
            metrics_names = self._params['metrics']
            metrics = [METRICS.get(m, m) for m in metrics_names]
            self._params['metrics'] = metrics

    def get_params(self):
        """Returns configured params"""
        return self._params

    def to_json(self):
        """Returns JSON for this object"""
        return self._json

    @staticmethod
    def from_json(json):
        """Constructs params from JSON"""
        return CompileParams(json)
