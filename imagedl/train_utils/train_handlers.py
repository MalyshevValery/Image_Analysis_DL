"""Train engine handlers"""
from typing import Mapping

import torch
from ignite.contrib.handlers import ProgressBar, TensorboardLogger
from ignite.engine import Events, Engine
from ignite.handlers import ModelCheckpoint, EarlyStopping
from ignite.metrics import Metric, RunningAverage
from torch import nn
from torch.optim import Optimizer
from torch.utils.data.dataloader import DataLoader

from imagedl import Config
from imagedl.data import Split
from imagedl.utility_config import EPOCH_BOUND, ALPHA
from .logger import info
from .metric_handling import metrics_to_str


def train_handlers(config: Config, trainer: Engine, val_eval: Engine,
                   model: nn.Module, optimizer: Optimizer,
                   split: Split, val_dl: DataLoader,
                   tb_logger: TensorboardLogger) -> ProgressBar:
    """Handlers for train process: metrics, latest and best model checkpoints
    and early stopping if requested"""
    metrics: Mapping[str, Metric]
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
    def log_validation_results(engine: Engine) -> None:
        """Log validation"""
        val_eval.run(val_dl)
        val_metrics = val_eval.state.metrics
        epoch = engine.state.epoch
        print_str = metrics_to_str(metrics, val_metrics, config.legend,
                                   tb_logger, engine.state.epoch, 'validation_')
        print_str = f'Validation #{epoch} - ' + print_str
        info(print_str)

    dict_to_save = {
        'model': model, 'optimizer': optimizer,
        'split': split, 'trainer': trainer
    }
    trainer.add_event_handler(Events.EPOCH_COMPLETED,
                              ModelCheckpoint(str(config.job_dir), 'latest'),
                              dict_to_save)

    if eval_metric is not None:
        mul = 1
        if eval_metric[0] == '-':
            mul = -1
            eval_metric = eval_metric[1:]

        def score_function(engine: Engine) -> object:
            """Score function for model"""
            return mul * val_eval.state.metrics[eval_metric]

        best_checkpoint = ModelCheckpoint(str(config.job_dir), 'best',
                                          score_function=score_function,
                                          score_name=eval_metric)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, best_checkpoint,
                                  dict_to_save)
        if patience is not None:
            trainer.add_event_handler(Events.EPOCH_COMPLETED,
                                      EarlyStopping(patience, score_function,
                                                    trainer))
        if test_load_best:
            @trainer.on(Events.COMPLETED)
            def reload_best(engine: Engine) -> None:
                """Reloads best checkpoint"""
                filename = best_checkpoint.last_checkpoint
                info('Loading best model...')
                model.load_state_dict(torch.load(filename)['model'])
    return progress_bar
