"""Handlers and loggers"""
from typing import TypeVar

import torch
from ignite.contrib.handlers import TensorboardLogger
from ignite.contrib.handlers.tensorboard_logger import OutputHandler
from ignite.engine import Events, Engine
from ignite.handlers import global_step_from_engine
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import make_grid

from imagedl import Config
from imagedl.nn.metrics.metric import UpgradedMetric
from imagedl.utility_config import TB_ITER
from .data import prepare_batch

T_co = TypeVar('T_co', covariant=True)


def tensorboard_logger(trainer: Engine, val_eval: Engine, model: nn.Module,
                       config: Config, train_dl: DataLoader[T_co],
                       val_dl: DataLoader[T_co],
                       device: torch.device) -> TensorboardLogger:
    """Creates tensorboard logger and attaches it to trainer and val evaluator.
    Also plots graph into tensorboard"""
    metrics, *_, train_metric_names = config.test
    valid_tb_metrics = ['loss']
    for k, v in metrics.items():
        if isinstance(v, UpgradedMetric) and v.vis:
            continue
        valid_tb_metrics.append(k)
    if train_metric_names is not None:
        train_show = list(
            set(valid_tb_metrics).intersection(train_metric_names + ['loss']))
    else:
        train_show = valid_tb_metrics

    tb_logger = TensorboardLogger(log_dir=config.job_dir)
    handler = OutputHandler(tag="training", metric_names=train_show)
    tb_logger.attach(trainer, handler,
                     Events.ITERATION_COMPLETED(every=TB_ITER))

    handler = OutputHandler(tag="validation", metric_names=valid_tb_metrics,
                            global_step_transform=global_step_from_engine(
                                trainer))
    tb_logger.attach(val_eval, handler, Events.EPOCH_COMPLETED)

    batch = next(iter(train_dl))
    tb_logger.writer.add_graph(model, (prepare_batch(batch, device)[0],))

    @trainer.on(Events.EPOCH_COMPLETED)
    def show_images_tb(engine: Engine) -> None:
        """Plots image to TensorBoard"""
        inp, targets = prepare_batch(next(iter(val_dl)), device)
        model.eval()
        pred = model(inp)
        visualized = config.visualize(inp, targets, pred)
        if visualized is not None:
            for name in visualized:
                visualized[name] = visualized[name].permute(0, 3, 1, 2)
                tb_logger.writer.add_image(f'validation_{name}',
                                           make_grid(visualized[name]),
                                           engine.state.epoch)

    return tb_logger
