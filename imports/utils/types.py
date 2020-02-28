"""Types for data"""
from typing import TypeVar, Union, Callable, Sequence

import numpy as np

T = TypeVar('T')
R = TypeVar('R')
OneMany = Union[T, Sequence[T]]
OMArray = OneMany[np.ndarray]


def to_seq(obj: OneMany[T]) -> Sequence[T]:
    """Returns tuple from OneMany type"""
    if isinstance(obj, Sequence):
        return obj
    else:
        return [obj]


def apply_as_seq(obj: OneMany[T],
                 f: Callable[[Sequence[T]], Sequence[R]]) -> OneMany[R]:
    """Applies function f which works with tuples to OneMany obj"""
    if isinstance(obj, Sequence):
        return f(obj)
    else:
        return f([obj])[0]
