# functions to calculate acoustic indices given time audio

from typing import Tuple

import maad
import numpy as np

from src.tools.loader import AudioSegment


def temporal_entropy(segment: AudioSegment, frame_size: int = 512) -> float:
    x = segment.data
    result = maad.features.temporal_entropy(x, Nt=frame_size)
    if result is None:
        return 0.0
    return float(result)


def amplitude_median(segment: AudioSegment, frame_size: int = 512) -> float:
    x = segment.getWaveform()
    return maad.features.temporal_median(x, Nt=frame_size)


def background_noise(segment: AudioSegment) -> float:
    return segment.getNoise()


def signal_to_noise_ratio(segment: AudioSegment) -> float:
    x = segment.getWaveform()
    noise = segment.getNoise()
    return np.max(x) - noise


def acoustic_activity(
    segment: AudioSegment, db_threshold: int = 3, frame_size: int = 512
) -> float:
    x = segment.data
    frac, _, _ = maad.features.temporal_activity(
        x, dB_threshold=db_threshold, Nt=frame_size
    )
    return frac


def acoustic_event_proportion_and_duration(
    segment: AudioSegment, db_threshold: int = 3, frame_size: int = 512
):
    x = segment.data
    sr = segment.sr
    frac, dur, _, _ = maad.features.temporal_events(
        x, sr, dB_threshold=db_threshold, Nt=frame_size
    )
    return float(frac), dur
