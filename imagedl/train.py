"""Run test features net"""
from pathlib import Path
from typing import Dict

import pandas as pd
import torch
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, \
    OutputHandler, global_step_from_engine, OptimizerParamsHandler
from ignite.engine import create_supervised_trainer, \
    create_supervised_evaluator, Events, Engine
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.metrics import Loss, RunningAverage, Metric
from torch.utils.data import DataLoader

from imagedl.config import Config
from imagedl.data import Splitter, Split
from imagedl.data.datasets import SubDataset
from .utility_config import DEVICE, WORKERS


def metrics_to_str(metrics: Dict[str, float]) -> str:
    """Put metrics in string"""
    sorted_names = sorted(metrics.keys())
    res = []
    for s in sorted_names:
        res.append(f'{s}: {metrics[s]:.3f}')
    return ' '.join(res)


def train_run(config: Config, split: Split, job_dir: Path) -> pd.DataFrame:
    """Training procedure depending on split"""
    job_dir.mkdir(parents=True, exist_ok=False)
    model, optimizer_fn, criterion = config.model_config
    model.to(DEVICE)
    optimizer = optimizer_fn(model.parameters())

    epochs, batch_size, patience = config.train
    dataset, _, train_transform, test_transform, _ = config.data
    train_ds = SubDataset(dataset, split.train, transform=train_transform)
    val_ds = SubDataset(dataset, split.val, transform=test_transform)
    test_ds = SubDataset(dataset, split.test, transform=test_transform)
    print(f'Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}')

    train_dl = DataLoader(train_ds, batch_size, True, num_workers=WORKERS)
    val_dl = DataLoader(val_ds, batch_size, num_workers=WORKERS)
    test_dl = DataLoader(test_ds, batch_size, num_workers=WORKERS)

    config.show_samples(train_ds, job_dir)

    trainer = create_supervised_trainer(model, optimizer, criterion,
                                        device=DEVICE,
                                        output_transform=lambda x, y, y_pred,
                                                                loss: (
                                            y_pred, y, loss.item()))

    metrics: Dict[str, Metric]
    metrics, eval_metric = config.test

    metrics['loss'] = Loss(criterion,
                           output_transform=lambda data: (data[0], data[1]))
    val_evaluator = create_supervised_evaluator(model, metrics, DEVICE)

    RunningAverage(output_transform=lambda x: x[2]).attach(trainer, "loss")
    for k, metric in metrics.items():
        RunningAverage(metric).attach(trainer, k)

    progress_bar = ProgressBar(persist=True)
    progress_bar.attach(trainer, metric_names="all")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine: Engine) -> None:
        """Log validation"""
        val_evaluator.run(val_dl)
        val_metrics = val_evaluator.state.metrics
        epoch = engine.state.epoch
        progress_bar.log_message(
            f'Validation #{epoch} - ' + metrics_to_str(val_metrics))
        progress_bar.n = progress_bar.last_print_n = 0

    def score_function(engine: Engine) -> object:
        """Score function for model"""
        return val_evaluator.state.metrics[eval_metric]

    handlers = [
        EarlyStopping(patience, score_function, trainer),
        ModelCheckpoint(str(job_dir), 'best', score_function=score_function,
                        score_name=eval_metric),
        ModelCheckpoint(str(job_dir), 'latest')
    ]
    save_dicts = [
        None,
        {'model': model},
        {'model': model, 'optimizer': optimizer, 'trainer': trainer}
    ]
    for handler, dict_ in zip(handlers, save_dicts):
        if dict_ is not None:
            trainer.add_event_handler(Events.EPOCH_COMPLETED, handler, dict_)
        else:
            trainer.add_event_handler(Events.EPOCH_COMPLETED, handler)

    tb_logger = TensorboardLogger(log_dir=job_dir)
    tb_trainer_handlers = [
        OutputHandler(tag="training", metric_names="all"),
        OptimizerParamsHandler(optimizer),
    ]
    for handler in tb_trainer_handlers:
        tb_logger.attach(trainer, handler, Events.ITERATION_COMPLETED(every=1))
    tb_logger.attach(
        val_evaluator,
        log_handler=OutputHandler(tag="validation", metric_names="all",
                                  global_step_transform=global_step_from_engine(
                                      trainer)),
        event_name=Events.EPOCH_COMPLETED, )
    tb_logger.writer.add_graph(model, next(iter(train_dl))[0])

    trainer.run(val_dl, max_epochs=epochs)
    tb_logger.close()

    to_save = job_dir / 'test'
    to_save.mkdir(parents=True, exist_ok=True)
    model.eval()
    cur = 0
    for i, data in enumerate(test_dl):
        inp, out = data
        pred = model(inp.to(DEVICE)).cpu()
        config.save_result(split.test[cur:cur + batch_size], inp, out,
                           pred, to_save)
        cur += test_dl.batch_size

    test_evaluator = create_supervised_evaluator(model, metrics, DEVICE)
    test_evaluator.run(test_dl)
    df = pd.DataFrame(test_evaluator.state.metrics, index=[0])
    df.to_csv(f'{job_dir}/metrics.csv', index=False)
    progress_bar.log_message(
        f'Test - ' + metrics_to_str(test_evaluator.state.metrics))

    if DEVICE.type == '  cuda':
        torch.cuda.empty_cache()
    return df


def train(config: Config, train_: float, val: float, test: float,
          kfold: int = None) -> None:
    """Main train procedure"""
    dataset, groups, _, _, job_dir = config.data

    print(f'Total number of samples: {len(dataset)}')
    splitter = Splitter(total=len(dataset), group_labels=groups)

    if kfold is None:
        split = splitter.random_split((train_, val, test))
        train_run(config, split, job_dir)
    else:
        frames = []
        for i, split in enumerate(
                splitter.k_fold(val, kfold)):
            print(f'Fold {i + 1}')
            fold_job_dir = job_dir / f'Fold_{i + 1}'
            frames.append(train_run(config, split, fold_job_dir))
            print(frames[-1])
        print(frames)
        overall = pd.concat(frames, ignore_index=True)
        print('All folds')
        print(overall)
        overall.to_csv(f'{job_dir}/metrics.csv')
