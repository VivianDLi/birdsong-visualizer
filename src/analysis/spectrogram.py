# 2d array of floats to represent long-duration spectrogram result

from typing import Dict

import numpy as np

from src.tools.interfaces import ISpectrogram


class Spectrogram(ISpectrogram):
    def __init__(self, result: Dict[str, np.ndarray], sr: int):
        if result:
            shapes = set([arr.shape for arr in result.values()])
            if len(shapes) > 1:
                raise ValueError(
                    "Results do not have the same shape. %s" % (shapes)
                )
            self.shape = shapes.pop()
        self.result = result
        self.sr = sr

    def getResult(self, index: str) -> np.ndarray:
        if index not in self.result:
            raise ValueError(f"{index} not available to get.")
        return self.result[index]

    def getColorResult(self, index: str) -> np.ndarray:
        if index not in self.result:
            raise ValueError(f"{index} not available to get.")
        result = self.result[index]
        result = result - np.min(result)  # lower bound is 0
        result *= 255.0 / np.max(result)  # normalize to 0 to 255
        return result

    def addIndex(self, index: str, result: np.ndarray) -> None:
        if self.result and result.shape != self.shape:
            raise ValueError(
                "Result does not have the correct shape. Expected shape is %s."
                % (self.shape)
            )
        else:
            self.shape = result.shape
        self.result[index] = result
