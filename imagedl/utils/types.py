"""Types definition"""
from typing import Union, Tuple, Sequence, TypeVar, Callable, Mapping, Optional

from ignite.engine import Engine
from torch import device, Tensor
from torch.nn import Module
from torch.optim import Optimizer

T_co = TypeVar('T_co', covariant=True)
DataType = Union[Tensor, Sequence['DataType'], Mapping[str, 'DataType']]
Transform = Callable[[T_co], DataType]
MetricTransform = Transform[Tuple[DataType]]
UpdateFun = Callable[[Engine, Sequence[Tensor]], DataType]
PrepareBatch = Callable[[DataType, device, bool], DataType]
OutputTransform = Callable[
    [Tensor, Tensor, Tensor, Tensor], Tuple[Tensor, Tensor, float]]

OutTransform = Callable[[Tensor, Tensor, Tensor, Tensor], Tuple[Tensor, ...]]
UpdateFunConstructor = Callable[
    [Module, Optimizer, Union[Callable, Module],
     Optional[Union[str, device]], OutTransform, PrepareBatch], Callable]
