# functions to calculate acoustic indices given time audio

import maad
import numpy as np

from tools.loader import AudioSegment


def temporal_entropy(segment: AudioSegment, frame_size=512):
    x = segment.data
    return maad.features.temporal_entropy(x, Nt=frame_size)


def amplitude_median(segment: AudioSegment, frame_size=512):
    x = segment.getWaveform()
    return maad.features.temporal_median(x, Nt=frame_size)


def background_noise(segment: AudioSegment):
    return segment.getNoise()


def signal_to_noise_ratio(segment: AudioSegment):
    x = segment.getWaveform()
    noise = segment.getNoise()
    return np.max(x) - noise


def acoustic_activity(segment: AudioSegment, db_threshold=3, frame_size=512):
    x = segment.data
    frac, _, _ = maad.features.temporal_activity(
        x, dB_threshold=db_threshold, Nt=frame_size)
    return frac


def acoustic_event_proportion_and_duration(segment: AudioSegment, db_threshold=3, frame_size=512):
    x = segment.data
    sr = segment.sr
    frac, dur, _, _ = maad.features.temporal_events(
        x, sr, dB_threshold=db_threshold, Nt=frame_size)
    return frac, dur
