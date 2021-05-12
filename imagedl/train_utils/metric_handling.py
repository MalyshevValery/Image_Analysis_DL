"""Metric values and labels preprocessing"""
from typing import Dict, List, Union, Mapping

import torch
from ignite.contrib.handlers import TensorboardLogger
from ignite.metrics import Metric

from imagedl.nn.metrics.metric import UpgradedMetric
from imagedl.utils.plot_saver import PlotSave


def clean_metrics(metrics: Mapping[str, Metric],
                  metric_values: Dict[str, Union[float, torch.Tensor]],
                  legend: List[str]) -> Dict[str, float]:
    """Clean metrics from 2+D metrics and
    prettify output of tensor-like metrics"""
    # TODO: Give legend a use
    sorted_names = sorted(metrics.keys())
    res = {}
    for s in sorted_names:
        val = metric_values[s]
        if isinstance(metrics[s], UpgradedMetric) and metrics[s].vis:
            continue
        if isinstance(val, torch.Tensor):
            if val.shape == ():
                res[s] = val.item()
            elif len(val.shape) == 1:
                for i in range(len(val)):
                    res[f'{s}_{legend[i]}'] = val[i].item()
        else:
            res[s] = val
    res['loss'] = float(metric_values['loss'])
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

    clean_values = clean_metrics(metrics, metric_values, legend)
    res = []
    for s in sorted(clean_values.keys()):
        res.append(f'{s}: {clean_values[s]:.3f}')
    return ' '.join(res)
