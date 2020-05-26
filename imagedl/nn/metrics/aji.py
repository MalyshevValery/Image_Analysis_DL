"""Confusion matrix metric"""
from typing import Tuple, Dict
import ignite
import torch
from imagedl.data.datasets.abstract import Transform


class AggregatedJaccardIndex(ignite.metrics.Metric):
    """
    Confusion matrix with quantifying

    :param output_transform: Transform for ignite.engine output before applying
        this metric
    """

    def __init__(self, output_transform: Transform = lambda x: x):
        self._updates = 0
        self._apply_reset = False
        self._iou = 0.0
        super().__init__(output_transform=output_transform)

    def reset(self) -> None:
        """Resets the metric"""
        self._apply_reset = True

    def _reset(self) -> None:
        self._iou = 0.0
        self._updates = 1

    def update(self, output: Tuple[torch.Tensor, torch.Tensor]) -> None:
        """Updates the metric"""
        logits_all, targets_all = self._prepare(output)
        for i in range(logits_all.shape[0]):
            logits, targets = logits_all[i], targets_all[i]
            inst_un, inst_cnt = targets.unique(return_counts=True)
            if inst_un[0] == 0:
                inst_un = inst_un[1:]
                inst_cnt = inst_cnt[1:]

            n_inst = len(inst_un)
            n_pred = logits.max()
            if n_inst == 0 and n_pred == 0:
                self._iou += 1.0
                continue
            if n_inst == 0 or n_pred == 0:
                continue

            pred_un, pred_cnt = logits.unique(return_counts=True)  # TODO: unique for predictions
            if pred_un[0] == 0:
                pred_un = pred_un[1:]
                pred_cnt = pred_cnt[1:]
            best_inter = torch.zeros((n_inst,))
            best_union = torch.zeros((n_inst,))
            pairs = torch.zeros((n_inst,), dtype=torch.uint8)
            for i, inst in enumerate(inst_un):
                inter_ind, inter_val = logits[torch.where(targets == inst)].unique(
                    return_counts=True)  # TODO: optimize logits background
                if inter_ind[0] == 0:
                    inter_ind = inter_ind[1:]
                    inter_val = inter_val[1:]
                if len(inter_ind) == 0:
                    best_union[i] = inst_cnt[i]
                    continue
                idx = torch.argmax(inter_val)
                pairs[i] = inter_ind[idx] - 1
                best_inter[i] = inter_val[idx]
                best_union[i] = inst_cnt[i] + pred_cnt[pairs[i].item()] - best_inter[i]
            union_unpaired = pred_cnt.sum() - pred_cnt[pairs.unique().long()].sum()
            self._iou += best_inter.sum() / (best_union.sum() + union_unpaired)

    def compute(self) -> float:
        """Metric aggregation"""
        assert self._updates > 0
        return self._iou / self._updates

    def _prepare(self, data: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        logits, targets = data
        logits = logits.cpu()
        targets = targets.cpu()
        if len(logits.shape) - len(targets.shape) == 1:
            targets = targets.unsqueeze(1)
        targets = targets.long()
        self._updates += 1
        return logits, targets
