from typing import Dict, List, Union, Mapping

import torch
from ignite.contrib.handlers import TensorboardLogger
from ignite.metrics import Metric

from imagedl.nn.metrics.metric import UpgradedMetric
from imagedl.utils.plot_saver import PlotSave


def clean_metrics(metrics: Mapping[str, Metric],
                  metric_values: Dict[str, Union[float, torch.Tensor]],
                  legend: List[str]) -> Dict[str, float]:
    sorted_names = sorted(metric_values.keys())
    res = {}
    for s in sorted_names:
        if s != 'loss' and isinstance(metrics[s], UpgradedMetric):
            if metrics[s].vis:
                continue
        if isinstance(metric_values[s], torch.Tensor):
            if metric_values[s].shape == ():
                res[s] = metric_values[s].item()
            elif len(metric_values[s].shape) == 1:
                for i in range(len(metric_values[s])):
                    res[f'{s}_{i}'] = metric_values[s][i].item()
        else:
            res[s] = metric_values[s]
    return res


def metrics_to_str(metrics: Mapping[str, Metric],
                   metric_values: Dict[str, Union[float, torch.Tensor]],
                   legend: List[str], tb_logger: TensorboardLogger,
                   epoch: int, prefix: str = '') -> str:
    """Put metrics in string"""
    sorted_names = sorted(metric_values.keys())
    for s in sorted_names:
        if s == 'loss':
            continue
        if isinstance(metrics[s], UpgradedMetric) and metrics[s].vis:
            with PlotSave(prefix + s, tb_logger, epoch):
                metrics[s].visualize(metric_values[s], legend)

    metric_values = clean_metrics(metrics, metric_values, legend)
    res = []
    for s in sorted(metric_values.keys()):
        try:
            res.append(f'{s}: {metric_values[s]:.3f}')
        except TypeError:
            pass
    return ' '.join(res)
