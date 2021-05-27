"""Sharpness Aware Minimization training procedure"""
from typing import Sequence, Union, Tuple, Callable, Optional

import torch
from ignite.engine import Engine
from torch import Tensor

from imagedl.nn.optim import SAM
from imagedl.utils.types import OutTransform, PrepareBatch


def sam_update_function(model: torch.nn.Module,
                        optimizer: SAM,
                        loss_fn: Union[Callable, torch.nn.Module],
                        device: Optional[Union[str, torch.device]],
                        output_transform: OutTransform,
                        prepare_batch: PrepareBatch) -> Callable:
    """Get SAM update function with first and second steps"""

    def sam_update(_: Engine,
                   batch: Sequence[Tensor]) -> Tuple[Tensor, ...]:
        """Update function for SAM"""
        model.train()
        optimizer.zero_grad()
        x, y = prepare_batch(batch, device, False)
        y_pred: Tensor = model(x)
        loss: Tensor = loss_fn(y_pred, y)
        loss.backward()
        optimizer.first_step()

        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.second_step(zero_grad=True)
        return output_transform(x, y, y_pred, loss)

    return sam_update
