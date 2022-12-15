# coordinate multiple analyzers in parallel

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List
import numpy as np
import pandas as pd
import itertools

from .analyzer import Analyzer
from .spectrogram import Spectrogram
from src.tools.interfaces import (
    ICoordinator,
    ISpectrogram,
    IAudioStream,
    IAudioSegment,
)


class AnalysisCoordinator(ICoordinator):
    def __init__(self, stream: IAudioStream):
        self.stream = stream
        self.spectrogram = Spectrogram({}, sr=stream.sr)

    def calculateSegment(self, i: int, *indices: str) -> ISpectrogram:
        segment = next(itertools.islice(self.stream, i, None))
        analyzer = Analyzer(segment)
        result, log = analyzer.calculateIndices(*indices)
        for (index, exc) in log:
            print(
                "Segment starting at %r generated an exception for index %s: %s"
                % (
                    self.stream.segmentToTimestamp(i),
                    index,
                    exc,
                ),
                flush=True,
            )
        if not hasattr(self.spectrogram, "shape"):
            self.spectrogram.shape = (
                self.stream.getNumberOfSegments(),
                len(result[indices[0]]),
            )
        self.spectrogram.addSegment(i, result)
        return self.spectrogram

    def calculateIndices(self, *indices: str) -> ISpectrogram:
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
        uncalculated_indices = [
            index
            for index in indices
            if index not in self.spectrogram.getIndices()
        ]
        if any([i not in supported_indices for i in uncalculated_indices]):
            raise ValueError("Unsupported acoustic indices were specified.")

        results = {
            index: np.zeros((self.stream.getNumberOfSegments(), 256))
            for index in uncalculated_indices
        }
        analyzers = [Analyzer(segment) for segment in self.stream]
        with ProcessPoolExecutor() as executor:
            futures_to_segment = {
                executor.submit(
                    analyzer.calculateIndices, *uncalculated_indices
                ): i
                for i, analyzer in enumerate(analyzers)
            }
            for future in as_completed(futures_to_segment):
                segment_number = futures_to_segment[future]
                try:
                    result, log = future.result()
                    for index in result:
                        results[index][segment_number] = result[index]
                    for (index, exc) in log:
                        print(
                            "Segment starting at %r generated an exception for index %s: %s"
                            % (
                                self.stream.segmentToTimestamp(segment_number),
                                index,
                                exc,
                            )
                        )
                except Exception as exc:
                    print(
                        "Segment starting at %r generated an exception: %s"
                        % (
                            self.stream.segmentToTimestamp(segment_number),
                            exc,
                        ),
                        flush=True,
                    )
        for index, result in results.items():
            self.spectrogram.addIndex(index, result)
        return self.spectrogram

    def loadIndices(self, path: str) -> List[str]:
        _, ext = os.path.splitext(path)
        if ext.lower() != ".csv":
            raise ValueError("file is not a .csv: %s" % (path))
        if not os.path.exists(path):
            raise ValueError("file doesn't exist: %s" % (path))
        df = pd.read_csv(path, header=[0, 1], index_col=0)
        if df.index.nlevels != 1 or df.columns.nlevels != 2:
            raise ValueError(
                "Dataframe obtained from .csv at %s has the wrong format. \
                The correct format should be frequencies as rows and (time indices, acoustic indices) as columns in a multi-level index."
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
        return new_indices

    def saveIndices(self, path: str) -> None:
        _, ext = os.path.splitext(path)
        if ext.lower() != ".csv":
            raise ValueError("file is not a .csv: %s" % (path))
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
