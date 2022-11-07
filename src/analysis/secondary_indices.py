# functions to calculate acoustic indices based on other calculated values

import maad

from analysis.temporal_indices import amplitude_median, temporal_entropy
from analysis.spectral_indices import spectral_entropy
from tools.loader import AudioSegment


def acoustic_richness_index(segment: AudioSegment):
    Ht = temporal_entropy(segment)
    M = amplitude_median(segment)
    ar = maad.features.acoustic_richness_index([Ht], [M])
    return ar[0]


def acoustic_entropy(segment: AudioSegment):
    Ht = temporal_entropy(segment)
    Hf = spectral_entropy(segment)
    return Ht * Hf
