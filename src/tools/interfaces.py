# define class interfaces

from abc import abstractmethod
from typing import Iterable, List, Tuple, Union, Dict
from typing_extensions import Protocol

import numpy as np


class IAudioSegment(Protocol):
    data: np.ndarray
    sr: int
    denoise: bool

    @abstractmethod
    def getWaveform(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def getNoise(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def getSpectrogram(self) -> Tuple[np.ndarray, List[float], List[float]]:
        raise NotImplementedError


class IAudioStream(Iterable[IAudioSegment], Protocol):
    file: str
    time_limits: Tuple[float, float]
    file_duration: float
    segment_duration: float

    @abstractmethod
    def createStream(
        self,
        start_time: float,
        end_time: float,
        segment_duration: Union[float, None],
    ) -> "IAudioStream":
        raise NotImplementedError

    @abstractmethod
    def createSTFT(self, n_fft: int, hop_length: int) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def play(
        self, offset: float = 0, duration: Union[float, None] = None
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        raise NotImplementedError

    def getNumberOfSegments(self) -> int:
        # ceiling division
        return int(-(self.file_duration // -self.segment_duration))

    def segmentToTimestamp(self, segment_number: int) -> float:
        return segment_number * self.segment_duration


class ISpectrogram(Protocol):
    result: Dict[str, np.ndarray]

    @abstractmethod
    def getResult(self, index: str) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def addIndex(self, index: str, result: np.ndarray) -> None:
        raise NotImplementedError

    def getIndices(self) -> List[str]:
        return list(self.result.keys())


class ICoordinator(Protocol):
    stream: IAudioStream
    current_indices: List[str]
    spectrogram: ISpectrogram

    @abstractmethod
    def calculateIndex(self, index: str) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def calculateIndices(self) -> ISpectrogram:
        raise NotImplementedError

    @abstractmethod
    def getSTFT(self) -> np.ndarray:
        raise NotImplementedError

    def getSpectrogram(self) -> ISpectrogram:
        return self.spectrogram


class IAnalyzer(Protocol):
    segment: IAudioSegment

    @abstractmethod
    def calculateIndex(self, index: str) -> np.ndarray:
        raise NotImplementedError
