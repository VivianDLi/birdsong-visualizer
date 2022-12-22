# define class interfaces

from abc import abstractmethod
from typing import Iterable, List, Optional, Tuple, Union, Dict
from typing_extensions import Protocol

import numpy as np
import threading


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


class IPlaybackThread(Protocol):
    playback_time: float

    @abstractmethod
    def stop(self):
        raise NotImplementedError


class IAudioStream(Iterable[IAudioSegment], Protocol):
    file: str
    sr: int
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
    ) -> IPlaybackThread:
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        raise NotImplementedError

    def getNumberOfSegments(self) -> int:
        # ceiling division
        return int(-(self.file_duration // -self.segment_duration))

    def getDuration(self) -> float:
        return self.time_limits[1] - self.time_limits[0]

    def segmentToTimestamp(self, segment_number: int) -> float:
        return segment_number * self.segment_duration


class ISpectrogram(Protocol):
    shape: Tuple[int, int]
    result: Dict[str, np.ndarray]
    sr: int

    @abstractmethod
    def getResult(self, index: str) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def getColorResult(self, index: str) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def addSegment(self, i: int, result: Dict[str, np.ndarray]) -> None:
        raise NotImplementedError

    @abstractmethod
    def addIndex(self, index: str, result: np.ndarray) -> None:
        raise NotImplementedError

    def getShape(self) -> Tuple[int, int]:
        return self.shape

    def getIndices(self) -> List[str]:
        return list(self.result.keys())

    def getFrequencies(self) -> List[float]:
        return list(
            np.linspace(0, self.sr // 2, self.shape[1], endpoint=False)
        )


class ICoordinator(Protocol):
    stream: IAudioStream
    spectrogram: ISpectrogram

    @abstractmethod
    def calculateSegment(self, i: int, *indices: str) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def calculateIndices(self, *indices: str) -> ISpectrogram:
        raise NotImplementedError

    @abstractmethod
    def getSTFT(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def loadIndices(self, path: str) -> ISpectrogram:
        raise NotImplementedError

    @abstractmethod
    def saveIndices(self, path) -> None:
        raise NotImplementedError

    def getSpectrogram(self) -> ISpectrogram:
        return self.spectrogram


class IAnalyzer(Protocol):
    segment: IAudioSegment

    @abstractmethod
    def calculateIndices(self, *indices: str) -> Dict[str, np.ndarray]:
        raise NotImplementedError
