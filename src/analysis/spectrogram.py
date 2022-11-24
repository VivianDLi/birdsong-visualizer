# 2d array of floats to represent long-duration spectrogram result

from typing import Dict

import numpy as np

from src.tools.interfaces import ISpectrogram


class Spectrogram(ISpectrogram):
    def __init__(self, result: Dict[str, np.ndarray]):
        self.result = result

    def getResult(self, index: str) -> np.ndarray:
        if index not in self.result:
            raise ValueError(f"{index} not available to get.")
        return self.result[index]

    def addIndex(self, index: str, result: np.ndarray) -> None:
        self.result[index] = result
