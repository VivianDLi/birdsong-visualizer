# coordinate multiple analyzers in parallel

from multiprocessing import Pool
from typing import List
import numpy as np

from .analyzer import Analyzer
from .result import Result
from src.tools.loader import AudioStream


class AnalysisCoordinator:
    def __init__(self, stream: AudioStream, indices: List[str]):
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
        self.indices = indices
        self.analyzers = [
            Analyzer(segment, indices, seg_num=i)
            for i, segment in enumerate(stream)
        ]

    def calculateIndices(self) -> Result:
        def collect_result(result):
            i, dict_result = result
            print(i)
            for index, values in dict_result.items():
                results[index][i] = values

        results = {
            index: np.zeros(self.stream.getNumSegments())
            for index in self.indices
        }
        with Pool() as pool:
            for analyzer in self.analyzers:
                pool.apply_async(
                    analyzer.calculateIndices, callback=collect_result
                )
            pool.close()
            pool.join()
        return Result(results, self.stream.segment_length)
