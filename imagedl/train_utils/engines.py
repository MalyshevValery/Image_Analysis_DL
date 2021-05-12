"""Create training and evaluation engines"""
from typing import Tuple, Any

import numpy as np
import pandas as pd
import torch
from ignite.contrib.handlers import TensorboardLogger
from ignite.engine import (create_supervised_evaluator,
                           create_supervised_trainer, Engine)
from ignite.metrics import Loss
from torch import Tensor, nn
from torch.utils.data.dataloader import DataLoader

from imagedl import Config
from imagedl.config import TestConfig
from imagedl.data import Split
from imagedl.nn.update_funs import update_functions
from .data import prepare_batch
from .logger import info
from .metric_handling import metrics_to_str, clean_metrics


def evaluator(test_config: TestConfig, criterion: nn.Module, model: nn.Module,
              device: torch.device) -> Engine:
    """Create evaluator for validation"""
    metrics, eval_metric, *_ = test_config
    metrics['loss'] = Loss(criterion,
                           output_transform=lambda data: (data[0], data[1]))
    val_evaluator = create_supervised_evaluator(model, metrics, device,
                                                prepare_batch=prepare_batch)
    return val_evaluator


def evaluate(config: Config, test_dl: DataLoader, test_split: np.ndarray,
             criterion: nn.Module, model: nn.Module,
             device: torch.device, tb_logger: TensorboardLogger,
             engine: Engine) -> pd.DataFrame:
    """Test evaluator"""
    metrics, eval_metric, *_ = config.test
    metrics['loss'] = Loss(criterion,
                           output_transform=lambda x: (x[0], x[1]))
    test_evaluator = create_supervised_evaluator(model, metrics, device)
    test_evaluator.run(test_dl)
    metric_values = test_evaluator.state.metrics
    cleaned_metrics = clean_metrics(metrics, metric_values, config.legend)
    df = pd.DataFrame(cleaned_metrics, index=[0])
    df.to_csv(f'{config.job_dir}/metrics.csv', index=False)
    info(f'Test - ' + metrics_to_str(metrics, test_evaluator.state.metrics,
                                     config.legend, tb_logger,
                                     engine.state.epoch + 1, 'test_'))

    to_save = config.job_dir / 'test'
    to_save.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        model.eval()
        batch_size = test_dl.batch_size
        for i, data in enumerate(test_dl):
            inp, out = prepare_batch(data, device)
            pred = model(inp)
            if isinstance(pred, torch.Tensor):
                pred = pred.cpu()
            else:
                pred = [p.cpu() for p in pred]
            img = config.visualize(inp, out, pred)
            if img is None:
                break
            indexes = test_split[i * batch_size:(i + 1) * batch_size]
            config.save_sample(img, to_save, indexes)
    return df


def create_trainer(config: Config, device: torch.device, split: Split,
                   own_split: bool) -> Tuple[Any, ...]:
    """Create training engine & load checkpoint"""
    ret_type = Tuple[Tensor, Tensor, float]

    def output_transform(x: Tensor, y: Tensor,
                         y_pred: Tensor, loss: Tensor) -> ret_type:
        """What trainer returns to metrics at each step"""
        return y_pred, y, loss.item()

    model, optimizer_fn, criterion, checkpoint = config.model_config
    model = model.to(device)
    optimizer = optimizer_fn(model.parameters())
    if optimizer.__class__ in update_functions:
        update_function = update_functions[optimizer.__class__]
        update = update_function(model, optimizer, criterion, device,
                                 output_transform, prepare_batch)
        trainer = Engine(update)
    else:
        trainer = create_supervised_trainer(model, optimizer, criterion, device,
                                            prepare_batch=prepare_batch,
                                            output_transform=output_transform)
    if checkpoint is not None:
        info(f'Resume from {checkpoint}')
        obj = torch.load(str(checkpoint))
        model.load_state_dict(obj['model'])
        optimizer.load_state_dict(obj['optimizer'])
        trainer.load_state_dict(obj['trainer'])
        if not own_split:
            split = Split.load_state_dict(obj['split'])
    return model, optimizer, criterion, split, trainer
