"""Run test features net"""
from pathlib import Path
from typing import Dict

import click
import pandas as pd
import torch
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.handlers.tensorboard_logger import *
from ignite.engine import create_supervised_trainer, \
    create_supervised_evaluator, Events
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.metrics import Loss, RunningAverage
from torch import optim
from torch.utils.data import DataLoader

from imagedl.data import Splitter, Split
from imagedl.data.datasets import SubDataset
from scripts.cam_seg_config import CAMSegConfig


def metrics_to_str(metrics: Dict[str, float]) -> str:
    """Put metrics in string"""
    sorted_names = sorted(metrics.keys())
    res = []
    for s in sorted_names:
        res.append(f'{s}: {metrics[s]:.3f}')
    return ' '.join(res)


def train(config: CAMSegConfig, split: Split, job_dir: Path) -> pd.DataFrame:
    """Training procedure depending on split"""
    job_dir.mkdir(parents=True, exist_ok=False)
    model = config.model.to(device=config.device)
    optimizer = optim.Adam(model.parameters())
    criterion = config.criterion

    train_ds = SubDataset(config.dataset, split.train,
                          transform=config.train_transform)
    val_ds = SubDataset(config.dataset, split.val,
                        transform=config.test_transform)
    test_ds = SubDataset(config.dataset, split.test,
                         transform=config.test_transform)

    train_dl = DataLoader(train_ds, config.batch_size, True,
                          num_workers=config.num_workers)
    val_dl = DataLoader(val_ds, config.batch_size,
                        num_workers=config.num_workers)
    test_dl = DataLoader(test_ds, config.batch_size,
                         num_workers=config.num_workers)

    config.show_samples(train_ds, job_dir)

    trainer = create_supervised_trainer(model, optimizer, criterion,
                                        device=config.device,
                                        output_transform=lambda x, y, y_pred,
                                                                loss: (
                                            y_pred, y, loss.item()))

    metrics = config.metrics
    metrics['loss'] = Loss(criterion)
    val_evaluator = create_supervised_evaluator(model, metrics, config.device)

    RunningAverage(output_transform=lambda x: x[2]).attach(trainer, "loss")
    for k, metric in config.metrics.items():
        RunningAverage(metric).attach(trainer, k)

    pbar = ProgressBar(persist=True)
    pbar.attach(trainer, metric_names="all")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        val_evaluator.run(val_dl)
        val_metrics = val_evaluator.state.metrics
        epoch = engine.state.epoch
        pbar.log_message(
            f'Validation #{epoch} - ' + metrics_to_str(val_metrics))
        pbar.n = pbar.last_print_n = 0

    score_function = lambda engine: val_evaluator.state.metrics[
        config.eval_metric]
    trainer.add_event_handler(Events.EPOCH_COMPLETED,
                              EarlyStopping(config.patience, score_function,
                                            trainer))
    trainer.add_event_handler(Events.EPOCH_COMPLETED,
                              ModelCheckpoint(str(job_dir), 'best',
                                              score_function=score_function,
                                              score_name=config.eval_metric),
                              {'model': model})
    trainer.add_event_handler(Events.EPOCH_COMPLETED,
                              ModelCheckpoint(str(job_dir), 'latest'),
                              {'model': model, 'optimizer': optimizer,
                               'trainer': trainer})

    tb_logger = TensorboardLogger(log_dir=job_dir)
    tb_logger.attach(
        trainer,
        log_handler=OutputHandler(tag="training", metric_names="all"),
        event_name=Events.ITERATION_COMPLETED)
    tb_logger.attach(
        val_evaluator,
        log_handler=OutputHandler(
            tag="validation",
            metric_names="all",
            global_step_transform=global_step_from_engine(trainer)),
        event_name=Events.EPOCH_COMPLETED, )

    tb_logger.attach(
        trainer, log_handler=OptimizerParamsHandler(optimizer),
        event_name=Events.ITERATION_COMPLETED(every=100)
    )

    print(model)
    tb_logger.writer.add_graph(model, next(iter(train_dl))[0])

    trainer.run(val_dl, max_epochs=config.epochs)
    tb_logger.close()

    to_save = job_dir / 'test'
    to_save.mkdir(parents=True, exist_ok=True)
    model.eval()
    cur = 0
    for i, data in enumerate(test_dl):
        inp, out = data
        pred = model(inp.to(config.device)).cpu()
        config.save_result(split.test[cur:cur + config.batch_size], inp, out,
                           pred, to_save)
        cur += test_dl.batch_size

    test_evaluator = create_supervised_evaluator(model, metrics, config.device)
    test_evaluator.run(test_dl)
    df = pd.DataFrame(test_evaluator.state.metrics, index=[0])
    df.to_csv(f'{job_dir}/metrics.csv', index=False)
    pbar.log_message(f'Test - ' + metrics_to_str(test_evaluator.state.metrics))

    if config.device.type == '  cuda':
        torch.cuda.empty_cache()
    return df


@click.command()
@click.option('--train', '-t', 'train_', default=0.8, help='Training size',
              type=float)
@click.option('--val', '-v', default=0.1, help='Validation size', type=float)
@click.option('--test', '-s', default=0.1, help='Test size', type=float)
@click.option('--kfold', '-k', help='Number of folds', type=int)
def main(train_: float, val: float, test: float, kfold: int = None,
         level: int = 5) -> None:
    """Main program"""
    config = CAMSegConfig()
    print(f'Total number of samples: {config.dataset}')
    splitter = Splitter(total=len(config.dataset), group_labels=config.groups)

    if kfold is None:
        split = splitter.random_split((train_, val, test))
        train(config, split, config.job_dir)
    else:
        frames = []
        for i, split in enumerate(
                splitter.k_fold(val, kfold)):
            print(f'Fold {i + 1}')
            fold_job_dir = config.job_dir / f'Fold_{i + 1}'
            frames.append(train(config, split, fold_job_dir))
            print(frames[-1])
        print(frames)
        overall = pd.concat(frames, ignore_index=True)
        print('All folds')
        print(overall)
        overall.to_csv(f'{config.job_dir}/metrics.csv')


if __name__ == '__main__':
    main()
