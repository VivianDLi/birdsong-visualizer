# coordinate multiple analyzers in parallel

from concurrent.futures import ProcessPoolExecutor, as_completed
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
        results = {
            index: np.zeros((self.stream.getNumSegments(), 256))
            for index in self.indices
        }

        with ProcessPoolExecutor() as executor:
            futures_to_segment = {
                executor.submit(analyzer.calculateIndices): i
                for i, analyzer in enumerate(self.analyzers)
            }
            for future in as_completed(futures_to_segment):
                segment_number = futures_to_segment[future]
                try:
                    i, dict_result = future.result()
                    for index, values in dict_result.items():
                        results[index][i] = np.array(values)
                except Exception as exc:
                    print(
                        "Segment starting at %r generated an exception: %s"
                        % (self.stream.segmentToTimestamp(segment_number), exc)
                    )
        return Result(results, self.stream.segment_length)
