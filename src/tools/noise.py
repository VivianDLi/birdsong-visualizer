# remove background noise of audio

from itertools import Counter

import numpy as np

# using adaptive level equalization algorithm from Towsey(2013)
# note: data is a 1D array scaled between -1 and 1 for audio amplitude
def waveform_denoise(data, frame_size = 512, filter_window = 3, sd_count = 0.1):
    # set constants
    min_env_dB = -60
    noise_threshold_dB = 10
    num_bins = 100
    upper_mode_bound = int(num_bins * 0.95)
    bin_width = noise_threshold_dB / num_bins
    histogram = np.zeros(num_bins)
    # get signal envelope (average of frames)
    envelope = [max(abs(data[i:i+frame_size])) for i in range(0, len(data), frame_size)]
    # convert to decibel (20 * log_10(signal))
    db_signal = [20 * np.log10(x) for x in envelope]
    # get minimum dBs
    min_dB = min(np.min(db_signal), min_env_dB)
    # populate histogram
    bg_threshold = min_dB + noise_threshold_dB
    bg_signal = db_signal[db_signal >= min_dB | db_signal <= bg_threshold]
    h_indices = [min(num_bins - 1, max(0, int((x - min_dB) / bin_width))) for x in bg_signal]
    counts = Counter(h_indices)
    histogram = [counts[i] for i in range(0, num_bins)]
    # smooth histogram
    cumsum = np.cumsum(histogram)
    smoothed = (cumsum - np.concatenate(np.zeros(filter_window), cumsum)[0:len(cumsum)]) / np.concatenate(np.arange(0, filter_window), filter_window * np.ones(len(cumsum) - filter_window))
    # calculate mode and std
    mode_index = min(np.argmax(smoothed), upper_mode_bound)
    smoothed_cumsum = np.cumsum(smoothed[0:mode_index + 1])
    threshold_sum = smoothed_cumsum[-1] * 0.68 # one std. dev.
    std_index = np.argmax(np.cumsum(smoothed[mode_index: 0: -1]) > threshold_sum)
    noise_mode = min_dB + ((mode_index + 1) * bin_width)
    noise_std = (mode_index - std_index) * bin_width
    # calculate background dB threshold
    noise_threshold_dB = noise_mode + (noise_std * sd_count)
    noise_threshold = 10 ** (noise_threshold_dB / 20) # noise dB scaled to data
    # denoise data
    return [max(x - noise_threshold, 0) if x > 0 else min(x + noise_threshold, 0) for x in data] # move noise_threshold closer to 0
    
# using adaptive level equalization algorithm from Towsey(2013)
# note: data is a 2D array 
def spectrogram_denoise(data):
    pass