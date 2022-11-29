# coordinate multiple analyzers in parallel

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List
import numpy as np
import pandas as pd

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
        self.spectrogram = Spectrogram({}, sr=stream.sr)
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

    def loadIndices(self, path: str) -> ISpectrogram:
        df = pd.read_csv(path, header=[0, 1], index_col=0)
        if df.index.nlevels != 1 or df.columns.nlevels != 2:
            raise ValueError(
                "Dataframe obtained from .csv at %s has the wrong format. The correct format should be frequencies as rows and (time indices, acoustic indices) as columns in a multi-level index."
                % (path)
            )
        new_indices: list[str] = df.columns.levels[1]  # type: ignore
        for index in new_indices:
            result = (
                df.loc[:, pd.IndexSlice[:, index]]  # type: ignore
                .droplevel(level=1, axis=1)
                .to_numpy()
            ).transpose()
            self.spectrogram.addIndex(index, result)
        return self.spectrogram

    def saveIndices(self, path: str) -> None:
        acoustic_indices = self.spectrogram.getIndices()
        time_indices = [
            self.stream.segmentToTimestamp(t)
            for t in range(self.stream.getNumberOfSegments())
        ]
        freq_indices = self.spectrogram.getFrequencies()
        row_index = pd.Index(freq_indices, name="frequencies")
        col_index = pd.MultiIndex.from_product(
            [time_indices, acoustic_indices], names=["time", "index"]
        )
        df = pd.DataFrame(0, index=row_index, columns=col_index)
        for index in acoustic_indices:
            result = pd.DataFrame(
                self.spectrogram.getResult(index).transpose(),
                index=row_index,
                columns=pd.MultiIndex.from_product(
                    [time_indices, [index]], names=["time", "index"]
                ),
            )
            df.loc[:, pd.IndexSlice[:, index]] = result  # type: ignore
        df.to_csv(path)

    def getSTFT(self, n_fft: int = 2048, hop_length: int = 1024) -> np.ndarray:
        return self.stream.createSTFT(n_fft, hop_length)
