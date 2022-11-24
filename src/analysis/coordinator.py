# coordinate multiple analyzers in parallel

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List
import numpy as np

from .analyzer import Analyzer
from .spectrogram import Spectrogram
from src.tools.interfaces import ICoordinator, ISpectrogram, IAudioStream


class AnalysisCoordinator(ICoordinator):
    def __init__(self, stream: IAudioStream, indices: List[str]):
        supported_indices = [
            "Ht",
            "M",
            "BgN",
            "SNR",
            "AcAct",
            "AEFrac",
            "AEDur",
            "Hf",
            "HfVar",
            "HfMax",
            "SpDiv",
            "SpAct",
            "ACI",
            "AEI",
            "BioI",
            "LFreqCov",
            "MFreqCov",
            "HFreqCov",
            "NDSI",
            "ARI",
            "H",
        ]
        if len(indices) > 3:
            raise ValueError(
                "Only up to three acoustic indices should be specified."
            )
        if any([i not in supported_indices for i in indices]):
            raise ValueError("Unsupported acoustic indices were specified.")
        self.stream = stream
        self.current_indices = indices
        self.spectrogram = Spectrogram({})
        self._analyzers = [Analyzer(segment) for segment in stream]

    def calculateIndex(self, index: str) -> np.ndarray:
        try:
            return self.spectrogram.getResult(index)
        except:
            results = np.empty((self.stream.getNumberOfSegments(), 256))
            with ProcessPoolExecutor() as executor:
                futures_to_segment = {
                    executor.submit(analyzer.calculateIndex, index): i
                    for i, analyzer in enumerate(self._analyzers)
                }
                for future in as_completed(futures_to_segment):
                    segment_number = futures_to_segment[future]
                    try:
                        result = future.result()
                        results[segment_number] = result
                    except Exception as exc:
                        print(
                            "Segment starting at %r generated an exception for index %s: %s"
                            % (
                                self.stream.segmentToTimestamp(segment_number),
                                index,
                                exc,
                            )
                        )
                        results[segment_number] = np.zeros(256)
            self.spectrogram.addIndex(index, results)
            return results

    def calculateIndices(self) -> ISpectrogram:
        for index in self.current_indices:
            self.calculateIndex(index)
        return self.spectrogram

    def getSTFT(self, n_fft: int = 2048, hop_length: int = 1024) -> np.ndarray:
        return self.stream.createSTFT(n_fft, hop_length)
