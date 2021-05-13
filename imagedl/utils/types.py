"""Types definition"""
from typing import Union, Tuple, Sequence, TypeVar, Callable, Mapping

from ignite.engine import Engine
from torch import device, Tensor
from torch.nn import Module
from torch.optim import Optimizer

T_co = TypeVar('T_co', covariant=True)
DataType = Union[Tensor, Sequence['DataType'], Mapping[str, 'DataType']]

UpdateFun = Callable[[Engine, Sequence[Tensor]], DataType]
PrepareBatch = Callable[[DataType, device, bool], DataType]
OutputTransform = Callable[
    [Tensor, Tensor, Tensor, Tensor], Tuple[Tensor, Tensor, float]]
UpdateFunConstructor = Callable[
    [Module, Optimizer, Module, device, OutputTransform,
     PrepareBatch], UpdateFun]
