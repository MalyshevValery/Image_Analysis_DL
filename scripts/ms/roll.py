"""Roll function"""
import numpy as np


def roll(a: np.ndarray, sz: int, step: int) -> np.ndarray:
    """Adds 3rd and 4th dimensions as tiles

    :param a: array to roll
    :param sz: size of a tile
    :param step: step between tiles
    :return:
    """
    shape = ((a.shape[0] - sz) // step + 1,
             (a.shape[1] - sz) // step + 1,
             sz, sz)
    strides = (a.strides[0] * step, a.strides[1] * step) + a.strides[-2:]
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
