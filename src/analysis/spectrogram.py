# 2d array of floats to represent long-duration spectrogram result

from typing import Dict, List, Optional, Tuple

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
        if np.max(result) != 0:
            result *= 1.0 / np.max(result)  # normalize to 0 to 1 for kivy RGB
        return result

    def addSegment(self, i: int, result: Dict[str, np.ndarray]) -> None:
        if i >= self.shape[0]:
            raise ValueError(
                "Index is out of bounds. Expected indices are below %s. Current index is %s."
                % (self.shape[0], i)
            )
        for index in result:
            if self.result and result[index].shape != (self.shape[1],):
                raise ValueError(
                    "Result does not have the correct shape. Expected shape is %s. Result shape is %s."
                    % ((self.shape[1],), result[index].shape)
                )
            if index in self.result:
                self.result[index][i] = result[index]
            else:
                self.result[index] = self._createZeroArray()
                self.result[index][i] = result[index]

    def addIndex(self, index: str, result: np.ndarray) -> None:
        if not hasattr(self, "shape") and result is not None:
            self.shape = result.shape
        if self.result and result.shape != self.shape:
            raise ValueError(
                "Result does not have the correct shape. Expected shape is %s. Result shape is %s."
                % (self.shape, result.shape)
            )
        self.result[index] = result

    def _createZeroArray(self):
        return np.zeros(shape=self.shape)
