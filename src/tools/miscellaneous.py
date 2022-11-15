# common mathematical functions

from typing import List, TypeVar
import numpy as np

T = TypeVar('T', int, float)


def moving_average(data: List[T], window_size: int) -> List[T]:
    cumsum = np.cumsum(data)
    smoothed = (cumsum - np.concatenate(np.zeros(window_size), cumsum)[0:len(cumsum)]) / np.concatenate(
        np.arange(0, window_size), window_size * np.ones(len(cumsum) - window_size))
    return smoothed
