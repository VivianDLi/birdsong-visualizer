# class for calculating acoustic indices given an audio segment

from analysis.temporal_indices import *
from analysis.spectral_indices import *
from analysis.secondary_indices import *


class Analyzer:
    def __init__(self, segment: AudioSegment, indices):
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
        self.index_functions = set([index_mapping[index] for index in indices])
        self.indices = indices
        self.segment = segment

    def calculateIndices(self):
        pass
