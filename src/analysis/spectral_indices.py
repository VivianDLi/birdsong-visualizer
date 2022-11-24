# functions to calculate acoustic indices given spectral information

from typing import List, Tuple
import maad
import numpy as np

from src.tools.interfaces import IAudioSegment


def spectral_entropy(segment: IAudioSegment) -> Tuple[float, float, float]:
    S, _, fn = segment.getSpectrogram()
    result = maad.features.spectral_entropy(S, fn, flim=(482, 8820))
    if result is not None:
        av: float
        var: float
        maxima: float
        av, var, _, maxima, _, _ = result
    else:
        av, var, maxima = 0.0, 0.0, 0.0
    return av, var, maxima


def spectral_diversity(segment: IAudioSegment) -> int:
    S, _, _ = segment.getSpectrogram()
    S_ampl = np.sqrt(S)
    S_grouped = [
        np.nanmean(
            np.pad(
                band,
                (0, 3 - band.size % 3),
                mode="constant",
                constant_values=np.NaN,
            ).reshape(-1, 3),
            axis=1,
        )
        for band in S_ampl
    ]
    S_binary = np.vstack(
        [
            _remove_isolated_peaks(np.where(band > 0.07, 1, 0))
            for band in S_grouped
        ]
    )
    return _cluster_peaks(S_binary)


def _remove_isolated_peaks(band: np.ndarray) -> np.ndarray:
    for i in range(1, len(band) - 1):
        if band[i - 1] == 0 and band[i + 1] == 0:
            band[i] = 0
    return band


def _cluster_peaks(S_binary: np.ndarray) -> int:
    training_set: np.ndarray = S_binary[np.count_nonzero(S_binary, axis=1) > 2]
    if len(training_set) < 9:
        return 0
    initial_representatives = np.random.randint(len(training_set), size=2)
    clusters: List[int] = [
        initial_representatives[0],
        initial_representatives[1],
    ]
    cluster_sizes = np.array([1, 1])

    iterations = 0
    old_cluster_sizes = None
    while iterations < 20 and old_cluster_sizes != cluster_sizes:
        old_cluster_sizes = cluster_sizes
        similarity = np.zeros((len(clusters), len(training_set)))
        for i, vector in enumerate(training_set):
            if i not in clusters:
                similarity = np.zeros(len(clusters))
                for j, cluster in enumerate(clusters):
                    representative = training_set[cluster]
                    similarity[j] = np.sum(
                        np.array(representative) & np.array(vector)
                    ) / np.sum(np.array(representative) | np.array(vector))
                cluster = np.argmax(similarity)
                if similarity[cluster] > 0.15:
                    clusters.append(i)
                    np.append(cluster_sizes, 1)
                else:
                    cluster_sizes[cluster] += 1
        clusters = np.array(clusters)[cluster_sizes > 1]
        cluster_sizes = cluster_sizes[cluster_sizes > 1]
        iterations += 1
    cluster_sizes = cluster_sizes[cluster_sizes > 3]
    return len(cluster_sizes)


def spectral_activity(segment: IAudioSegment) -> np.ndarray:
    S, _, _ = segment.getSpectrogram()
    S_dB = maad.util.power2dB(S)
    frac, _, _ = maad.features.spectral_activity(S_dB)
    return frac


def acoustic_complexity_index(segment: IAudioSegment) -> np.ndarray:
    S, _, _ = segment.getSpectrogram()
    S_ampl = np.sqrt(S)
    _, bins, _ = maad.features.acoustic_complexity_index(S_ampl)
    return bins  # average across bins


def acoustic_evenness_index(segment: IAudioSegment) -> float:
    S, _, fn = segment.getSpectrogram()
    S_ampl = np.sqrt(S)
    return maad.features.acoustic_eveness_index(S_ampl, fn)


def bioacoustic_index(segment: IAudioSegment) -> float:
    S, _, fn = segment.getSpectrogram()
    S_ampl = np.sqrt(S)
    return maad.features.bioacoustics_index(S_ampl, fn, flim=(2000, 11000))


def frequency_band_cover(
    segment: IAudioSegment, db_threshold: int = 3
) -> Tuple[float, float, float]:
    S, _, fn = segment.getSpectrogram()
    S_dB = maad.util.power2dB(S)
    low, mid, high = maad.features.spectral_cover(
        S_dB,
        fn,
        dB_threshold=db_threshold,
        flim_LF=(0, 482),
        flim_MF=(482, 3500),
        flim_HF=(3500, 11000),
    )
    return low, mid, high


def normalized_difference_soundscape_index(segment: IAudioSegment) -> float:
    S, _, fn = segment.getSpectrogram()
    ndsi, _, _, _ = maad.features.soundscape_index(
        S, fn, flim_bioPh=(2000, 11000), flim_antroPh=(1000, 2000)
    )
    return ndsi
