# class for calculating acoustic indices given an audio segment

from typing import Dict, Callable, Iterable

from .temporal_indices import *
from .spectral_indices import *
from .secondary_indices import *
from src.tools.interfaces import IAnalyzer, IAudioSegment


class Analyzer(IAnalyzer):
    def __init__(self, segment: IAudioSegment):
        self.segment = segment
        self._index_mapping: Dict[str, Callable] = {
            "Ht": temporal_entropy,
            "M": amplitude_median,
            "BgN": background_noise,
            "SNR": signal_to_noise_ratio,
            "AcAct": acoustic_activity,
            "AEFrac": acoustic_event_proportion_and_duration,
            "AEDur": acoustic_event_proportion_and_duration,
            "Hf": spectral_entropy,
            "HfVar": spectral_entropy,
            "HfMax": spectral_entropy,
            "SpDiv": spectral_diversity,
            "SpAct": spectral_activity,
            "ACI": acoustic_complexity_index,
            "AEI": acoustic_evenness_index,
            "BioI": bioacoustic_index,
            "LFreqCov": frequency_band_cover,
            "MFreqCov": frequency_band_cover,
            "HFreqCov": frequency_band_cover,
            "NDSI": normalized_difference_soundscape_index,
            "ARI": acoustic_richness_index,
            "H": acoustic_entropy,
        }
        self._calculated: dict[Callable, np.ndarray] = {}

    def calculateIndex(self, index: str) -> np.ndarray:
        function = self._index_mapping[index]
        if function in self._calculated:
            return self._get_correct_result(index, self._calculated[function])
        result = function(self.segment)
        self._calculated[function] = result
        return self._get_correct_result(index, result)

    def _get_correct_result(self, index: str, result) -> np.ndarray:
        value = result
        if index == "AEFrac" or index == "Hf" or index == "LFreqCov":
            value = result[0]
        if index == "AEDur" or index == "HfVar" or index == "MFreqCov":
            value = result[1]
        if index == "HfMax" or index == "HFreqCov":
            value = result[2]
        if isinstance(value, Iterable):
            return np.array(value)
        else:
            return value * np.ones(256)
