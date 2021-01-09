import pandas as pd
import torch
from ignite.engine import create_supervised_evaluator, create_supervised_trainer
from ignite.metrics import Loss

from .data import prepare_batch
from .metric_handling import metrics_to_str, clean_metrics
from ..data import Split


def evaluator(test_config, criterion, model, device):
    metrics, eval_metric, *_ = test_config
    metrics['loss'] = Loss(criterion,
                           output_transform=lambda data: (data[0], data[1]))
    evaluator = create_supervised_evaluator(model, metrics, device,
                                            prepare_batch=prepare_batch,
                                            non_blocking=True)
    return evaluator


def evaluate(config, test_dl, test_split, criterion, progress_bar, model,
             device, tb_logger, engine):
    metrics, eval_metric, *_ = config.test
    metrics['loss'] = Loss(criterion,
                           output_transform=lambda data: (data[0], data[1]))
    test_evaluator = create_supervised_evaluator(model, metrics, device)
    test_evaluator.run(test_dl)
    metric_values = test_evaluator.state.metrics
    cleaned_metrics = clean_metrics(metrics, metric_values, config.legend)
    df = pd.DataFrame(cleaned_metrics, index=[0])
    df.to_csv(f'{config.job_dir}/metrics.csv', index=False)
    progress_bar.log_message(
        f'Test - ' + metrics_to_str(metrics, test_evaluator.state.metrics,
                                    config.legend, tb_logger,
                                    engine.state.epoch + 1, 'test_'))

    to_save = config.job_dir / 'test'
    to_save.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        model.eval()
        cur = 0
        for i, data in enumerate(test_dl):
            inp, out = prepare_batch(data, device)
            pred = model(inp)
            if isinstance(pred, torch.Tensor):
                pred = pred.cpu()
            else:
                pred = [p.cpu() for p in pred]
            config.save_sample(config.visualize(inp, out, pred), to_save,
                               test_split[cur:cur + test_dl.batch_size])
            cur += test_dl.batch_size
    return df


def create_trainer(config, device, split, own_split):
    model, optimizer_fn, criterion, checkpoint = config.model_config
    # if DISTRIBUTED is not None:
    #     model.to(DEVICE)
    #     model = DistributedDataParallel(model, DISTRIBUTED)
    optimizer = optimizer_fn(model.parameters())
    trainer = create_supervised_trainer(model, optimizer, criterion,
                                        prepare_batch=prepare_batch,
                                        device=device,
                                        output_transform=lambda x, y, y_pred,
                                                                loss: (
                                            y_pred, y, loss.item()))

    if checkpoint is not None:
        print(f'Resume from {checkpoint}')
        obj = torch.load(str(checkpoint))
        model.load_state_dict(obj['model'])
        optimizer.load_state_dict(obj['optimizer'])
        trainer.load_state_dict(obj['trainer'])
        if not own_split:
            split = Split.load_state_dict(obj['split'])

    return model, optimizer, criterion, split, trainer
