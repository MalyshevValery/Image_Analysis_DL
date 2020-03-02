"""Types for data"""
from typing import TypeVar, Union, Callable, Sequence

import numpy as np

T = TypeVar('T')
R = TypeVar('R')
OMArray = Union[np.ndarray, Sequence[np.ndarray]]


def to_seq(obj: Union[T, Sequence[T]]) -> Sequence[T]:
    """Returns tuple from OneMany type"""
    if isinstance(obj, Sequence):
        return obj
    else:
        return [obj]


def seq_apply(obj: Union[T, Sequence[T]],
              f: Callable[[Sequence[T]], Sequence[R]]) -> Union[R, Sequence[R]]:
    """Applies function f which works with tuples to OneMany obj"""
    if isinstance(obj, Sequence):
        return f(obj)
    else:
        return f([obj])[0]
