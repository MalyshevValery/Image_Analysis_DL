from typing import Dict, Union, List

import torch
from ignite.contrib.handlers import TensorboardLogger, ProgressBar
from ignite.contrib.handlers.tensorboard_logger import OutputHandler
from ignite.engine import Events
from ignite.handlers import global_step_from_engine, EarlyStopping, \
    ModelCheckpoint
from ignite.metrics import Metric, RunningAverage
from seaborn import heatmap
from torchvision.utils import make_grid

from imagedl.utility_config import ALPHA, EPOCH_BOUND, TB_ITER
from imagedl.utils.plot_saver import PlotSave
from .data import prepare_batch


def clean_metrics(metrics: Dict[str, Union[float, torch.Tensor]], legend: List[str]) -> Dict[str, float]:
    sorted_names = sorted(metrics.keys())
    res = {}
    for s in sorted_names:
        if isinstance(metrics[s], torch.Tensor):
            if metrics[s].shape == ():
                res[s] = metrics[s].item()
            elif len(metrics[s].shape) == 1:
                for i in range(len(metrics[s])):
                    res[f'{s}_{i}'] = metrics[s][i].item()
        else:
            res[s] = metrics[s]
    return res


def metrics_to_str(metrics: Dict[str, Union[float, torch.Tensor]], legend: List[str], tb_logger: TensorboardLogger,
                   epoch: int) -> str:
    """Put metrics in string"""
    sorted_names = sorted(metrics.keys())
    for s in sorted_names:
        if isinstance(metrics[s], torch.Tensor) and len(metrics[s].shape) == 2:
            with PlotSave(s, tb_logger, epoch):
                heatmap(metrics[s].cpu().numpy(), annot=True, fmt='.2f', xticklabels=legend, yticklabels=legend)

    metrics = clean_metrics(metrics, legend)
    sorted_names = sorted(metrics.keys())
    res = []
    for s in sorted_names:
        res.append(f'{s}: {metrics[s]:.3f}')
    return ' '.join(res)


def tensorboard_logger(trainer, val_eval, model, config, train_dl, val_dl,
                       device):
    tb_logger = TensorboardLogger(log_dir=config.job_dir)
    handler = OutputHandler(tag="training", metric_names="all")
    tb_logger.attach(trainer, handler, Events.ITERATION_COMPLETED(every=TB_ITER))

    handler = OutputHandler(tag="validation", metric_names="all",
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
        for name in visualized:
            visualized[name] = visualized[name].permute(0, 3, 1, 2)
            tb_logger.writer.add_image(f'validation_{name}',
                                       make_grid(visualized[name]),
                                       engine.state.epoch)

    return tb_logger


def train_handlers(config, trainer, val_eval, model, optimizer, split, val_dl, tb_logger):
    metrics: Dict[str, Metric]
    metrics, eval_metric = config.test

    RunningAverage(output_transform=lambda x: x[2], alpha=ALPHA,
                   epoch_bound=EPOCH_BOUND).attach(trainer, "loss")
    for k, metric in metrics.items():
        RunningAverage(metric, alpha=ALPHA,
                       epoch_bound=EPOCH_BOUND).attach(trainer, k)

    _, eval_metric = config.test
    *_, patience = config.train

    progress_bar = ProgressBar(persist=True)
    progress_bar.attach(trainer, metric_names="all")

    # if DISTRIBUTED is not None:
    #     @trainer.on(Events.EPOCH_STARTED)
    #     def set_epoch(engine):
    #         train_sampler.set_epoch(engine.state.epoch)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine) -> None:
        """Log validation"""
        val_eval.run(val_dl)
        val_metrics = val_eval.state.metrics
        epoch = engine.state.epoch
        progress_bar.log_message(
            f'Validation #{epoch} - ' + metrics_to_str(val_metrics, config.legend, tb_logger, engine.state.epoch))
        progress_bar.n = progress_bar.last_print_n = 0

    def score_function(engine) -> object:
        """Score function for model"""
        return val_eval.state.metrics[eval_metric]

    handlers = [
        EarlyStopping(patience, score_function, trainer),
        ModelCheckpoint(str(config.job_dir), 'best',
                        score_function=score_function,
                        score_name=eval_metric),
        ModelCheckpoint(str(config.job_dir), 'latest')
    ]
    dict_to_save = {
        'model': model,
        'optimizer': optimizer,
        'trainer': trainer,
        'split': split
    }
    save_dicts = [None, dict_to_save, dict_to_save]
    for handler, dict_ in zip(handlers, save_dicts):
        if dict_ is not None:
            trainer.add_event_handler(Events.EPOCH_COMPLETED, handler, dict_)
        else:
            trainer.add_event_handler(Events.EPOCH_COMPLETED, handler)
    return progress_bar
