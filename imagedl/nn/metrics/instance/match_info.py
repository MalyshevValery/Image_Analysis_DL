"""Instance Match Info"""
from typing import Tuple, NamedTuple

import torch
from ignite.metrics.metric import Metric, reinit__is_reduced

from imagedl.data.datasets.abstract import Transform


class ImageEvalResults(NamedTuple):
    """Complex results of one image"""
    pred_area: torch.Tensor
    pred_class: torch.Tensor

    target_area: torch.Tensor
    target_class: torch.Tensor

    target_to_pred: torch.Tensor
    pred_to_target: torch.Tensor
    ious: torch.Tensor
    inter: torch.Tensor
    union: torch.Tensor

    conf: torch.Tensor


class InstanceMatchInfo(Metric):
    """
    Confusion matrix with quantifying

    :param output_transform: Transform for ignite.engine output before applying
        this metric
    """

    def __init__(self, n_classes=None,
                 output_transform: Transform = lambda x: x,
                 use_confidence=False, device='cpu'):
        assert n_classes is None or n_classes > 1
        self._results = []
        self._device = device
        self._computed = None
        self._apply_reset = False
        self._n_classes = n_classes
        self._use_confidence = use_confidence
        super().__init__(output_transform=output_transform)

    @reinit__is_reduced
    def reset(self) -> None:
        """Resets the metric"""
        self._apply_reset = True

    def _reset(self) -> None:
        self._results.clear()
        self._apply_reset = False
        self._computed = None

    @reinit__is_reduced
    def update(self, output: Tuple[torch.Tensor, torch.Tensor]) -> None:
        """Updates the metric"""
        preds_inst, preds_class, preds_conf, targets_inst, targets_class = self._prepare(
            output)
        for i in range(preds_inst.shape[0]):
            pred_inst, target_inst = preds_inst[i], targets_inst[i]

            # calc unique ids and areas (sorted for implicity, seems its True by default)
            target_un, target_area = target_inst.unique(return_counts=True,
                                                        sorted=True)
            if target_un[0] == 0:
                target_un = target_un[1:].long()
                target_area = target_area[1:]
            pred_un, pred_area = pred_inst.unique(return_counts=True,
                                                  sorted=True)
            if pred_un[0] == 0:
                pred_un = pred_un[1:].long()
                pred_area = pred_area[1:]
            pred_mapping = {pred_un[i].item(): torch.tensor(i) for i in
                            range(len(pred_un))}

            # prepare confidence
            if self._use_confidence:
                conf = torch.stack(
                    [torch.tensor(preds_conf[i][v.item()]) for v in
                     pred_un]).to(target_area.device)
            else:
                conf = None

            n_inst = len(target_un)
            n_pred = len(pred_un)

            pred_class = torch.zeros(n_pred, dtype=torch.long,
                                     device=pred_area.device)
            target_class = torch.zeros(n_inst, dtype=torch.long,
                                       device=target_area.device)
            if self._n_classes != None:
                pred_class_map = preds_class[i]
                target_class_map = targets_class[i]
                for i, pred_id in enumerate(pred_un):
                    # calculate classes
                    inst_classes = pred_class_map[
                        torch.where(pred_inst == pred_id)]
                    class_un, class_area = inst_classes.unique(
                        return_counts=True, sorted=True)
                    if class_un[0] == 0:
                        class_un = class_un[1:]
                        class_area = class_area[1:]
                    if len(class_area) == 0:
                        pred_class[i] = 0
                        continue
                    idx = class_area.argmax()
                    pred_class[i] = class_un[idx] - 1

            # values to compute
            target_to_pred = torch.zeros(n_inst, dtype=torch.long,
                                         device=target_area.device)
            pred_to_target = torch.full((n_pred,), -1, dtype=torch.long,
                                        device=target_area.device)
            ious = torch.zeros(n_inst, device=target_area.device)
            inter = torch.zeros(n_inst, device=target_area.device)
            union = torch.zeros(n_inst, device=target_area.device)

            # matching loop
            for i, target_id in enumerate(target_un):
                inst_idx = torch.where(target_inst == target_id)
                # class for instance
                if self._n_classes is not None:
                    inst_classes = target_class_map[
                        torch.where(target_inst == target_id)]
                    class_un, class_area = inst_classes.unique(
                        return_counts=True, sorted=True)
                    if class_un[0] == 0:
                        class_un = class_un[1:]
                        class_area = class_area[1:]

                    idx = class_area.argmax()
                    target_class[i] = class_un[idx] - 1

                inter_ind, inter_val = pred_inst[inst_idx].unique(
                    return_counts=True, sorted=True)
                if inter_ind[0] == 0:
                    inter_ind = inter_ind[1:]
                    inter_val = inter_val[1:]

                isin = pred_un.view(1, -1).eq(inter_ind.view(-1, 1)).sum(0) == 1
                pred_inter_area = pred_area[isin]
                if len(inter_ind) == 0:
                    target_to_pred[i] = -1
                    ious[i] = inter[i] = 0.0
                    union[i] = target_area[i]
                    continue

                union_inst = target_area[i] + pred_inter_area - inter_val
                iou_inst = inter_val.float() / union_inst

                indices = torch.argsort(iou_inst, descending=True)
                sorted_idx = torch.stack(
                    [pred_mapping[id_.item()] for id_ in inter_ind[indices]])
                available = torch.where(pred_to_target[sorted_idx] == -1)[0]
                if len(available) == 0:
                    target_to_pred[i] = -1
                    ious[i] = inter[i] = 0.0
                    union[i] = target_area[i]
                    continue

                sel = available[0]
                idx = sorted_idx[sel]
                target_to_pred[i] = idx
                pred_to_target[idx] = i
                ious[i] = iou_inst[indices[sel]]
                inter[i] = inter_val[indices[sel]]
                union[i] = union_inst[indices[sel]]
            self._results.append(ImageEvalResults(
                pred_area.float(), pred_class,
                target_area.float(), target_class,
                target_to_pred, pred_to_target, ious, inter.float(),
                union.float(),
                conf
            ))

    def compute(self, full=False) -> float:
        """Metric aggregation"""
        assert len(self._results) > 0 or self._computed is not None
        if len(self._results) == 0:
            return self._computed
        to_cat = self._results
        self._results = []
        if self._computed is not None:
            to_cat.insert(0, self._computed)

        t_to_p = [c.target_to_pred for c in to_cat]
        p_to_t = [c.pred_to_target for c in to_cat]
        tlen = torch.tensor([0] + [len(t) for t in t_to_p[:-1]])
        plen = torch.tensor([0] + [len(p) for p in p_to_t[:-1]])
        tlen = tlen.cumsum(0)
        plen = plen.cumsum(0)
        for i in range(len(t_to_p)):
            t_to_p[i][t_to_p[i] != -1] += plen[i]
            p_to_t[i][p_to_t[i] != -1] += tlen[i]

        self._computed = ImageEvalResults(
            pred_area=torch.cat([c.pred_area for c in to_cat]),
            pred_class=torch.cat([c.pred_class for c in to_cat]),
            target_area=torch.cat([c.target_area for c in to_cat]),
            target_class=torch.cat([c.target_class for c in to_cat]),
            target_to_pred=torch.cat(t_to_p),
            pred_to_target=torch.cat(p_to_t),
            ious=torch.cat([c.ious for c in to_cat]),
            inter=torch.cat([c.inter for c in to_cat]),
            union=torch.cat([c.union for c in to_cat]),
            conf=torch.cat([c.conf for c in to_cat]) if self._use_confidence else None
        )

        if full:
            return self._computed
        else:
            return (self._computed.target_to_pred != -1).float().mean().item()

    def _prepare(self, data):
        logits, targets = data
        logits_inst, logits_class, logits_conf = self.__unpack(logits)
        targets_inst, targets_class = self.__unpack(targets, True)

        targets_inst = self.__prepare_tensor(logits_inst, targets_inst)
        if logits_class != None:
            assert targets_class.max().item() <= self._n_classes
            assert logits_class.shape[1] == self._n_classes or \
                   logits_class.shape[1] == 1
            targets_class = self.__prepare_tensor(logits_class, targets_class)

        if self._apply_reset:
            self._reset()
        return logits_inst, logits_class, logits_conf, targets_inst, targets_class

    @property
    def n_classes(self):
        return self._n_classes

    def __prepare_tensor(self, pred, target):
        if len(pred.shape) - len(target.shape) == 1:
            target = target.unsqueeze(1)
        target = target.long()
        return target

    def __unpack(self, pack, is_target=False):
        if isinstance(pack, torch.Tensor):
            assert self._n_classes is None
            return (pack, None) if is_target else (pack, None, None)
        inst = pack[0].detach().to(self._device)
        clazz = None
        conf = None
        c = 1
        if self._n_classes is not None:
            clazz = pack[c].detach().to(self._device)
            c += 1
        if not is_target and self._use_confidence:
            conf = pack[c]
        return (inst, clazz) if is_target else (inst, clazz, conf)
