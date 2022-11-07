# common mathematical functions

import numpy as np


def moving_average(data, window_size):
    cumsum = np.cumsum(data)
    smoothed = (cumsum - np.concatenate(np.zeros(window_size), cumsum)[0:len(cumsum)]) / np.concatenate(
        np.arange(0, window_size), window_size * np.ones(len(cumsum) - window_size))
    return smoothed
