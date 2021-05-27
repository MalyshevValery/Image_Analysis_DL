"""Sharpness-Aware minimization"""
from typing import Iterator, Type, Optional, Callable

import torch
from torch.nn import Parameter
from torch.optim import Optimizer


class SAM(Optimizer):
    """Sharpness-Aware Minimization"""

    def __init__(self, params: Iterator[Parameter],
                 base_optimizer: Type[Optimizer], rho: float = 0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        """First optimization step (Statistics)"""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        """Second step (Forward pass)"""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update
        if zero_grad:
            self.zero_grad()

    def step(self,
             closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Not implemented"""
        raise NotImplementedError(
            "You should first call `first_step` and the `second_step`")

    def _grad_norm(self) -> torch.Tensor:
        shared_device = self.param_groups[0]["params"][
            0].device  # put everything on the same device, helps parallelism
        norm: torch.Tensor = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm
