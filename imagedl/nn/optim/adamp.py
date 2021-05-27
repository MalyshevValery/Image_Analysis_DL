"""
AdamP
Copyright (c) 2020-present NAVER Corp.
MIT license
"""

import math
from typing import Tuple, Iterator, Callable, Optional

import torch
import torch.nn.functional as f
from torch import Tensor
from torch.nn import Parameter
from torch.optim.optimizer import Optimizer


def _channel_view(x: Tensor) -> Tensor:
    return x.view(x.size()[0], -1)


def _layer_view(x: Tensor) -> Tensor:
    return x.view(1, -1)


def _cosine_similarity(x: Tensor, y: Tensor, eps: float,
                       view_func: Callable[[Tensor], Tensor]) -> Tensor:
    x = view_func(x)
    y = view_func(y)
    return f.cosine_similarity(x, y, eps=eps).abs_()


def _projection(p: Tensor, grad: Tensor, perturb: float, delta: float,
                wd_ratio: float, eps: float) -> Tuple[float, float]:
    wd = 1.0
    expand_size = [-1] + [1] * (len(p.shape) - 1)
    for view_func in [_channel_view, _layer_view]:

        cosine_sim = _cosine_similarity(grad, p.data, eps, view_func)

        if cosine_sim.max() < delta / math.sqrt(view_func(p.data).size()[1]):
            p_n = p.data / view_func(p.data).norm(dim=1).view(
                expand_size).add_(eps)
            perturb -= p_n * view_func(p_n * perturb).sum(dim=1).view(
                expand_size)
            wd = wd_ratio

            return perturb, wd

    return perturb, wd


class AdamP(Optimizer):
    """AdamP optimizer, use as regular Adam"""

    def __init__(self, params: Iterator[Parameter], lr: float = 1e-3,
                 betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0, delta: float = 0.1,
                 wd_ratio: float = 0.1, nesterov: bool = False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        delta=delta, wd_ratio=wd_ratio, nesterov=nesterov)
        super(AdamP, self).__init__(params, defaults)

    def step(self,
             closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Optimizer step"""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                beta1, beta2 = group['betas']
                nesterov = group['nesterov']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                # Adam
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(
                    group['eps'])
                step_size = group['lr'] / bias_correction1

                if nesterov:
                    perturb = (beta1 * exp_avg + (1 - beta1) * grad) / denom
                else:
                    perturb = exp_avg / denom

                # Projection
                wd_ratio = 1.0
                if len(p.shape) > 1:
                    perturb, wd_ratio = _projection(p, grad, perturb,
                                                    group['delta'],
                                                    group['wd_ratio'],
                                                    group['eps'])

                # Weight decay
                if group['weight_decay'] > 0:
                    p.data.mul_(
                        1 - group['lr'] * group['weight_decay'] * wd_ratio)

                # Step
                p.data.add_(perturb, alpha=-step_size)

        return loss
