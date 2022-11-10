# class for calculating acoustic indices given an audio segment

from analysis.temporal_indices import *
from analysis.spectral_indices import *
from analysis.secondary_indices import *
from tools.loader import AudioSegment


class Analyzer:
    def __init__(self, segment: AudioSegment, indices, seg_num):
        index_mapping = {
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
            "H": acoustic_entropy
        }
        self.index_functions = {index: index_function for index,
                                index_function in index_mapping.items() if index in indices}
        self.segment = segment
        self.seg_num = seg_num

    def calculateIndices(self):
        calculated = {}
        results = {}
        for index, function in self.index_functions:
            if function in calculated:
                results[index] = self._get_correct_result(
                    index, calculated[function])
            else:
                result = function(self.segment)
                calculated[function] = result
                results[index] = self._get_correct_result(index, result)
        return (self.seg_num, results)

    def _get_correct_result(index, result):
        if index == "AEFrac" or index == "Hf" or index == "LFreqCov":
            return result[0]
        if index == "AEDur" or index == "HfVar" or index == "MFreqCov":
            return result[1]
        if index == "HfMax" or index == "HFreqCov":
            return result[2]
        else:
            return result
