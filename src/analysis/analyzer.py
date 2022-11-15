# class for calculating acoustic indices given an audio segment

from functools import partial
from typing import Any, Dict, Callable

from .temporal_indices import *
from .spectral_indices import *
from .secondary_indices import *
from src.tools.loader import AudioSegment


class Analyzer:
    def __init__(self, segment: AudioSegment, indices: List[str], seg_num: int):
        index_mapping: Dict[str, Callable] = {
            "Ht": partial(temporal_entropy, segment),
            "M": partial(amplitude_median, segment),
            "BgN": partial(background_noise, segment),
            "SNR": partial(signal_to_noise_ratio, segment),
            "AcAct": partial(acoustic_activity, segment),
            "AEFrac": partial(acoustic_event_proportion_and_duration, segment),
            "AEDur": partial(acoustic_event_proportion_and_duration, segment),
            "Hf": partial(spectral_entropy, segment),
            "HfVar": partial(spectral_entropy, segment),
            "HfMax": partial(spectral_entropy, segment),
            "SpDiv": partial(spectral_diversity, segment),
            "SpAct": partial(spectral_activity, segment),
            "ACI": partial(acoustic_complexity_index, segment),
            "AEI": partial(acoustic_evenness_index, segment),
            "BioI": partial(bioacoustic_index, segment),
            "LFreqCov": partial(frequency_band_cover, segment),
            "MFreqCov": partial(frequency_band_cover, segment),
            "HFreqCov": partial(frequency_band_cover, segment),
            "NDSI": partial(normalized_difference_soundscape_index, segment),
            "ARI": partial(acoustic_richness_index, segment),
            "H": partial(acoustic_entropy, segment),
        }
        self.index_functions: Dict[str, Callable] = {
            index: index_function
            for index, index_function in index_mapping.items()
            if index in indices
        }
        self.seg_num = seg_num

    def calculateIndices(self):
        calculated: dict[Callable, Any] = {}
        results = {}
        for index, function in self.index_functions.items():
            if function in calculated:
                results[index] = self._get_correct_result(
                    index, calculated[function]
                )
            else:
                result = function()
                calculated[function] = result
                results[index] = self._get_correct_result(index, result)
        return (self.seg_num, results)

    def _get_correct_result(self, index: str, result):
        if index == "AEFrac" or index == "Hf" or index == "LFreqCov":
            return result[0]
        if index == "AEDur" or index == "HfVar" or index == "MFreqCov":
            return result[1]
        if index == "HfMax" or index == "HFreqCov":
            return result[2]
        else:
            return result
