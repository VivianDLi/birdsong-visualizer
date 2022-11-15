# functions to calculate acoustic indices based on other calculated values

from typing import List

import maad

from .temporal_indices import amplitude_median, temporal_entropy
from .spectral_indices import spectral_entropy
from src.tools.loader import AudioSegment


def acoustic_richness_index(segment: AudioSegment) -> float:
    Ht = temporal_entropy(segment)
    M = amplitude_median(segment)
    ar = maad.features.acoustic_richness_index([Ht], [M])
    return ar[0]


def acoustic_entropy(segment: AudioSegment) -> float:
    Ht = temporal_entropy(segment)
    Hf, _, _ = spectral_entropy(segment)
    return Ht * Hf
