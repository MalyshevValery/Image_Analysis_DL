"""Run test features net"""
from pathlib import Path
from typing import Dict, Union, Iterable

import pandas as pd
import torch
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, \
    OutputHandler, global_step_from_engine, OptimizerParamsHandler, WeightsHistHandler, WeightsScalarHandler, \
    GradsHistHandler, GradsScalarHandler
from ignite.engine import create_supervised_trainer, \
    create_supervised_evaluator, Events, Engine
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.metrics import Loss, RunningAverage, Metric
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from imagedl.config import Config
from imagedl.data import Splitter, Split
from imagedl.data.datasets import SubDataset
from .utility_config import DEVICE, WORKERS, ALPHA, EPOCH_BOUND, DISTRIBUTED
from ignite.utils import convert_tensor

import torch.distributed as distributed
import torch.utils.data.distributed as data_distributed
from torch.nn.parallel import DistributedDataParallel


def _prepare_batch(batch, device=None, non_blocking=False):
    new_batch = []
    for b in batch:
        if isinstance(b, torch.Tensor):
            new_batch.append(convert_tensor(b, device, non_blocking))
        elif isinstance(b, dict):
            new_b = {}
            for k in b.keys():
                new_b[k] = convert_tensor(b[k], device, non_blocking)
            new_batch.append(new_b)
        elif isinstance(b, Iterable):
            new_batch.append([convert_tensor(b_i, device, non_blocking) for b_i in b])
        else:
            raise ValueError('Wrong data format')

    return tuple(new_batch)


def metrics_to_str(metrics: Dict[str, Union[float, torch.Tensor]]) -> str:
    """Put metrics in string"""
    sorted_names = sorted(metrics.keys())
    res = []
    for s in sorted_names:
        if isinstance(metrics[s], torch.Tensor):
            if metrics[s].shape == ():
                res.append(f'{s}: {metrics[s]:.3f}')
            elif len(metrics[s].shape) == 1:
                res.append(f'{s}: {metrics[s]}')
        else:
            res.append(f'{s}: {metrics[s]:.3f}')
    return ' '.join(res)


def train_run(config: Config, split: Split, job_dir: Path) -> pd.DataFrame:
    """Training procedure depending on split"""
    job_dir.mkdir(parents=True, exist_ok=True)
    model, optimizer_fn, criterion, checkpoint = config.model_config
    if DISTRIBUTED is not None:
        model.to(DEVICE)
        model = DistributedDataParallel(model, DISTRIBUTED, output_device=DEVICE)
    optimizer = optimizer_fn(model.parameters())
    trainer = create_supervised_trainer(model, optimizer, criterion,
                                        device=DEVICE, prepare_batch=_prepare_batch, non_blocking=True,
                                        output_transform=lambda x, y, y_pred,
                                                                loss: (
                                            y_pred, y, loss.item()))

    if checkpoint is not None:
        print(f'Resume from {checkpoint}')
        obj = torch.load(str(checkpoint))
        model.load_state_dict(obj['model'])
        optimizer.load_state_dict(obj['optimizer'])
        trainer.load_state_dict(obj['trainer'])
        split = Split.load_state_dict(obj['split'])

    epochs, batch_size, patience = config.train
    dataset, _, train_transform, test_transform, _ = config.data
    train_ds = SubDataset(dataset, split.train, transform=train_transform)
    val_ds = SubDataset(dataset, split.val, transform=test_transform)
    test_ds = SubDataset(dataset, split.test, transform=test_transform)
    print(f'Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}')

    if DISTRIBUTED is not None:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
    else:
        train_sampler = None
    train_dl = DataLoader(train_ds, batch_size, num_workers=WORKERS, sampler=train_sampler,
                          shuffle=(train_sampler is None))
    val_dl = DataLoader(val_ds, batch_size, num_workers=WORKERS, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size, num_workers=WORKERS, shuffle=False)

    config.save_sample(config.visualize(*next(iter(train_dl))),
                       job_dir / 'train')

    metrics: Dict[str, Metric]
    metrics, eval_metric = config.test

    RunningAverage(output_transform=lambda x: x[2], alpha=ALPHA,
                   epoch_bound=EPOCH_BOUND).attach(trainer, "loss")
    for k, metric in metrics.items():
        RunningAverage(metric, alpha=ALPHA,
                       epoch_bound=EPOCH_BOUND).attach(trainer, k)

    metrics, eval_metric = config.test
    metrics['loss'] = Loss(criterion,
                           output_transform=lambda data: (data[0], data[1]))
    val_evaluator = create_supervised_evaluator(model, metrics, DEVICE, prepare_batch=_prepare_batch, non_blocking=True)

    progress_bar = ProgressBar(persist=True)
    progress_bar.attach(trainer, metric_names="all")

    if DISTRIBUTED is not None:
        @trainer.on(Events.EPOCH_STARTED)
        def set_epoch(engine):
            train_sampler.set_epoch(engine.state.epoch)

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

    batch = next(iter(train_dl))
    # tb_logger.writer.add_graph(model, _prepare_batch(batch, device=DEVICE)[0])

    @trainer.on(Events.EPOCH_COMPLETED)
    def show_images_tb(engine: Engine) -> None:
        """Plots image to TensorBoard"""
        inp, targets = _prepare_batch(next(iter(val_dl)), device=DEVICE)
        model.eval()
        pred = model(inp)
        visualized = config.visualize(inp, targets, pred)
        for name in visualized:
            visualized[name] = visualized[name].permute(0, 3, 1, 2)
            tb_logger.writer.add_image(f'validation_{name}', make_grid(visualized[name]), engine.state.epoch)

    trainer.run(train_dl, max_epochs=epochs)
    tb_logger.close()

    test_evaluator = create_supervised_evaluator(model, metrics, DEVICE)
    test_evaluator.run(test_dl)
    metrics = test_evaluator.state.metrics
    cleaned_metrics = {}
    for s in metrics.keys():
        if isinstance(metrics[s], torch.Tensor) and metrics[s].shape != ():
            continue
        cleaned_metrics[s] = metrics[s]
    df = pd.DataFrame(cleaned_metrics, index=[0])
    df.to_csv(f'{job_dir}/metrics.csv', index=False)
    progress_bar.log_message(
        f'Test - ' + metrics_to_str(test_evaluator.state.metrics))

    to_save = job_dir / 'test'
    to_save.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        model.eval()
        cur = 0
        for i, data in enumerate(test_dl):
            inp, out = data
            pred = model(inp.to(DEVICE))
            if isinstance(pred, torch.Tensor):
                pred = pred.cpu()
            else:
                pred = [p.cpu() for p in pred]
            config.save_sample(config.visualize(inp, out, pred), to_save,
                               split.test[cur:cur + batch_size])
            cur += test_dl.batch_size

    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
    return df


def train(config: Config, train_: float, val: float, test: float,
          kfold: int = None) -> None:
    """Main train procedure"""
    if DISTRIBUTED:
        torch.cuda.set_device(DEVICE)
        torch.distributed.init_process_group("nccl", init_method="file:///tmp/nccl_dist.dat",
                                             world_size=len(DISTRIBUTED),
                                             rank=0)

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
