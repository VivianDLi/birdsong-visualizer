# remove background noise of audio

from collections import Counter
from typing import List, Tuple

import numpy as np
import maad

from .miscellaneous import moving_average

# using adaptive level equalization algorithm from Towsey(2013)
# note: data is a 1D array of dB values converted from average signal amplitude


def waveform_denoise(data: List[float], frame_size: int = 512, filter_window: int = 3, sd_count: float = 0.1) -> Tuple[List[float], float]:
    # set constants
    min_env_dB = -60
    noise_threshold_dB = 10
    num_bins = 100
    upper_mode_bound = int(num_bins * 0.95)
    bin_width = noise_threshold_dB / num_bins
    # get signal envelope (average of frames)
    envelope = maad.sound.envelope(data, Nt=frame_size)
    # get minimum dBs
    min_dB = min(np.min(envelope), min_env_dB)
    # populate histogram
    bg_threshold = min_dB + noise_threshold_dB
    bg_signal = envelope[envelope >= min_dB | envelope <= bg_threshold]
    h_indices = [
        min(num_bins - 1, max(0, int((x - min_dB) / bin_width)))
        for x in bg_signal
    ]
    counts = Counter(h_indices)
    histogram = [counts[i] for i in range(0, num_bins)]
    # smooth histogram
    smoothed = moving_average(histogram, filter_window)
    # calculate mode and std
    mode_index = min(int(np.argmax(smoothed)), upper_mode_bound)
    smoothed_cumsum = np.cumsum(smoothed[0: mode_index + 1])
    threshold_sum = smoothed_cumsum[-1] * 0.68  # one std. dev.
    std_index = int(
        np.argmax(np.cumsum(smoothed[mode_index:0:-1]) > threshold_sum))
    noise_mode = min_dB + ((mode_index + 1) * bin_width)
    noise_std = (mode_index - std_index) * bin_width
    # calculate background dB threshold
    noise_threshold = noise_mode + (noise_std * sd_count)
    # denoise data
    return [max(x - noise_threshold, 0) for x in data], noise_threshold


# using adaptive level equalization algorithm from Towsey(2013)
# note: data is a 2D array of (frequency, frames) as power


def spectrogram_denoise(data: List[List[float]], filter_window: int = 5, sd_count: float = 0.1) -> List[List[float]]:
    # calculate thresholds
    thresholds = [
        _calculate_spectral_threshold(xs, filter_window, sd_count)
        for xs in data
    ]
    thresholds = moving_average(thresholds, filter_window)
    # subtract threshold values
    result = np.clip([np.array(xs) - thresholds[i]
                     for i, xs in enumerate(data)], 0, None)
    return result


# calculate spectral threshold per frequency bin for noise reduction


def _calculate_spectral_threshold(data: List[float], filter_window: int, sd_count: float) -> float:
    # define constants
    num_bins = int(len(data) / 8)
    upper_mode_bound = int(num_bins * 0.95)
    min_power = np.min(data)
    max_power = np.max(data)
    bin_width = (max_power - min_power) / num_bins
    # populate histogram
    h_indices = [
        min(num_bins - 1, max(0, int((x - min_power) / bin_width)))
        for x in data
    ]
    counts = Counter(h_indices)
    histogram = [counts[i] for i in range(0, num_bins)]
    # smooth histogram
    smoothed = moving_average(histogram, filter_window)
    # calculate mode and std
    mode_index = min(int(np.argmax(smoothed)), upper_mode_bound)
    smoothed_cumsum = np.cumsum(smoothed[0: mode_index + 1])
    threshold_sum = smoothed_cumsum[-1] * 0.68  # one std. dev.
    std_index = np.argmax(np.cumsum(smoothed[mode_index:0:-1]) > threshold_sum)
    noise_mode = min_power + ((mode_index + 1) * bin_width)
    noise_std = (mode_index - std_index) * bin_width
    # calculate threshold
    return noise_mode + (noise_std * sd_count)
