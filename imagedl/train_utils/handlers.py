import logging
from typing import Dict, Union, List

import torch
from ignite.contrib.handlers import TensorboardLogger, ProgressBar
from ignite.contrib.handlers.tensorboard_logger import OutputHandler
from ignite.engine import Events
from ignite.handlers import global_step_from_engine, EarlyStopping, \
    ModelCheckpoint
from ignite.metrics import Metric, RunningAverage
from torchvision.utils import make_grid

from imagedl.utility_config import ALPHA, EPOCH_BOUND, TB_ITER
from imagedl.utils.plot_saver import PlotSave
from .data import prepare_batch
from ..nn.metrics.metric import UpgradedMetric


def clean_metrics(metrics: [str, Metric],
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


def metrics_to_str(metrics: Dict[str, Metric],
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
        res.append(f'{s}: {metric_values[s]:.3f}')
    return ' '.join(res)


def tensorboard_logger(trainer, val_eval, model, config, train_dl, val_dl,
                       device):
    tb_logger = TensorboardLogger(log_dir=config.job_dir)
    handler = OutputHandler(tag="training", metric_names='all')
    tb_logger.attach(trainer, handler,
                     Events.ITERATION_COMPLETED(every=TB_ITER))
    metrics, *_ = config.test
    valid_tb_metrics = []
    for k, v in metrics.items():
        if isinstance(v, UpgradedMetric) and v.vis:
            continue
        valid_tb_metrics.append(k)
    handler = OutputHandler(tag="validation", metric_names=valid_tb_metrics,
                            global_step_transform=global_step_from_engine(
                                trainer))
    tb_logger.attach(val_eval, handler, Events.EPOCH_COMPLETED)

    batch = next(iter(train_dl))
    tb_logger.writer.add_graph(model, prepare_batch(batch, device)[0])

    @trainer.on(Events.EPOCH_COMPLETED)
    def show_images_tb(engine) -> None:
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


def train_handlers(config, trainer, val_eval, model, optimizer, split, val_dl,
                   tb_logger):
    metrics: Dict[str, Metric]
    metrics, eval_metric, test_load_best, train_metric_names = config.test

    RunningAverage(output_transform=lambda x: x[2], alpha=ALPHA,
                   epoch_bound=EPOCH_BOUND).attach(trainer, "loss")
    for k, metric in metrics.items():
        if train_metric_names is None or k in train_metric_names:
            RunningAverage(metric, alpha=ALPHA,
                           epoch_bound=EPOCH_BOUND).attach(trainer, k)

    *_, patience = config.train

    progress_bar = ProgressBar(persist=True)
    progress_bar.attach(trainer, metric_names="all")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine) -> None:
        """Log validation"""
        val_eval.run(val_dl)
        val_metrics = val_eval.state.metrics
        epoch = engine.state.epoch
        print_str = metrics_to_str(metrics, val_metrics, config.legend,
                                   tb_logger, engine.state.epoch, 'validation_')
        print_str = f'#{epoch} - ' + print_str
        logging.info(print_str)
        progress_bar.log_message('Validation ' + print_str)
        progress_bar.n = progress_bar.last_print_n = 0

    mul = 1 if eval_metric[0] != '-' else -1
    eval_metric = eval_metric[1:] if eval_metric[0] == '-' else eval_metric

    def score_function(engine) -> object:
        """Score function for model"""
        return mul * val_eval.state.metrics[eval_metric]

    handlers = [
        ModelCheckpoint(str(config.job_dir), 'best',
                        score_function=score_function,
                        score_name=eval_metric),
        ModelCheckpoint(str(config.job_dir), 'latest')
    ]
    if patience is not None:
        handlers.append(EarlyStopping(patience, score_function, trainer))

    dict_to_save = {
        'model': model,
        'optimizer': optimizer,
        'trainer': trainer,
        'split': split
    }
    save_dicts = [dict_to_save, dict_to_save, None]
    for handler, dict_ in zip(handlers, save_dicts):
        if dict_ is not None:
            trainer.add_event_handler(Events.EPOCH_COMPLETED, handler, dict_)
        else:
            trainer.add_event_handler(Events.EPOCH_COMPLETED, handler)

    @trainer.on(Events.COMPLETED)
    def reload_best(engine):
        """Reloads best checkpoint"""
        best_checkpoint_handler = handlers[1]
        filename = best_checkpoint_handler.last_checkpoint
        if test_load_best:
            print('Loading best model...')
            model.load_state_dict(torch.load(filename)['model'])

    return progress_bar
