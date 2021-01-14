from typing import Sequence, Union, Tuple, Any, Callable, Optional

import torch
from ignite.engine import Engine

from imagedl.nn.optim import SAM

RetType = Union[Any, Tuple[torch.Tensor]]


def sam_update_function(model: torch.nn.Module,
                        optimizer: SAM,
                        loss_fn: Union[Callable, torch.nn.Module],
                        device: Optional[Union[str, torch.device]],
                        output_transform: Callable,
                        prepare_batch: Callable) -> Callable:
    """Get SAM update function with first and second steps"""

    def sam_update(engine: Engine, batch: Sequence[torch.Tensor]) -> RetType:
        """Update function for SAM"""
        model.train()
        optimizer.zero_grad()
        x, y = prepare_batch(batch, device=device, non_blocking=False)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.first_step()

        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.second_step(zero_grad=True)
        return output_transform(x, y, y_pred, loss)

    return sam_update
