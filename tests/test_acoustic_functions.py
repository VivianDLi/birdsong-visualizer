import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal

from src.tools.loader import load_audio
from src.analysis.temporal_indices import *
from src.analysis.spectral_indices import *
from src.analysis.secondary_indices import *


class TestAcousticFunctions(unittest.TestCase):
    def assertArrayEqual(self, a, b, msg=None):
        try:
            assert_array_almost_equal(a, b)
        except AssertionError:
            raise self.failureException(msg)

    def setUp(self):
        self.stream = load_audio("./tests/test.wav")

        self.addTypeEqualityFunc(np.ndarray, self.assertArrayEqual)

    def test_temporal_entropy_returns_expected_result(self):
        expected_results = [1.0, 1.0, 1.0]
        for i, segment in enumerate(self.stream):
            result = temporal_entropy(segment)

            self.assertAlmostEqual(result, expected_results[i], places=2)

    def test_amplitude_median_returns_expected_result(self):
        expected_results = [89.535, 89.535, 89.535]
        for i, segment in enumerate(self.stream):
            result = amplitude_median(segment)

            self.assertAlmostEqual(result, expected_results[i], places=2)

    def test_background_noise_returns_expected_result(self):
        expected_results = [-49.45, -49.45, -49.45]
        for i, segment in enumerate(self.stream):
            result = background_noise(segment)

            self.assertAlmostEqual(result, expected_results[i], places=2)

    def test_signal_to_noise_ratio_returns_expected_result(self):
        expected_results = [138.985, 138.985, 138.985]
        for i, segment in enumerate(self.stream):
            result = signal_to_noise_ratio(segment)

            self.assertAlmostEqual(result, expected_results[i], places=2)

    def test_acoustic_activity_returns_expected_result(self):
        expected_results = [0.0, 0.0, 0.0]
        for i, segment in enumerate(self.stream):
            result = acoustic_activity(segment)

            self.assertAlmostEqual(result, expected_results[i], places=2)

    def test_acoustic_event_returns_expected_result(self):
        expected_proportions = [0.0, 0.0, 0.0]
        expected_durations = [0.0, 0.0, 0.0]
        for i, segment in enumerate(self.stream):
            frac, dur = acoustic_event_proportion_and_duration(segment)

            self.assertAlmostEqual(frac, expected_proportions[i], places=2)
            self.assertEqual(dur, expected_durations[i])

    def test_spectral_entropy_returns_expected_result(self):
        expected_averages = [0.891, 0.722, 0.891]
        expected_variances = [0.854, 0.794, 0.892]
        expected_maximums = [1.0, 0.869, 1.0]
        for i, segment in enumerate(self.stream):
            average, variance, maximum = spectral_entropy(segment)

            self.assertAlmostEqual(average, expected_averages[i], places=2)
            self.assertAlmostEqual(variance, expected_variances[i], places=2)
            self.assertAlmostEqual(maximum, expected_maximums[i], places=2)

    def test_spectral_diversity_returns_expected_result(self):
        expected_results = [0, 0, 0]
        for i, segment in enumerate(self.stream):
            result = spectral_diversity(segment)

            self.assertEqual(result, expected_results[i])

    def test_spectral_activity_returns_expected_result(self):
        expected_results = [[0.0] * 256, [0.0] * 256, [0.0] * 256]
        for i, segment in enumerate(self.stream):
            result = spectral_activity(segment)

            self.assertEqual(np.array(result), np.array(expected_results[i]))

    def test_acoustic_complexity_index_returns_expected_shape(self):
        for i, segment in enumerate(self.stream):
            result = acoustic_complexity_index(segment)

            self.assertEqual(np.array(result).shape, (256,))

    def test_acoustic_evenness_index_returns_expected_result(self):
        expected_results = [0.942, 0.929, 0.943]
        for i, segment in enumerate(self.stream):
            result = acoustic_evenness_index(segment)

            self.assertAlmostEqual(result, expected_results[i], places=2)

    def test_bioacoustic_index_returns_expected_result(self):
        expected_results = [15.143, 16.833, 14.183]
        for i, segment in enumerate(self.stream):
            result = bioacoustic_index(segment)

            self.assertAlmostEqual(result, expected_results[i], places=2)

    def test_frequency_band_cover_returns_expected_result(self):
        expected_lows = [0.0, 0.0, 0.0]
        expected_mediums = [0.0, 0.0, 0.0]
        expected_highs = [0.0, 0.0, 0.0]
        for i, segment in enumerate(self.stream):
            low, medium, high = frequency_band_cover(segment)

            self.assertAlmostEqual(low, expected_lows[i], places=2)
            self.assertAlmostEqual(medium, expected_mediums[i], places=2)
            self.assertAlmostEqual(high, expected_highs[i], places=2)

    def test_normalized_difference_soundscape_index_returns_expected_result(
        self,
    ):
        expected_results = [-0.999, -0.999, -0.999]
        for i, segment in enumerate(self.stream):
            result = normalized_difference_soundscape_index(segment)

            self.assertAlmostEqual(result, expected_results[i], places=2)

    def test_acoustic_richness_index_returns_expected_result(self):
        expected_results = [1.0, 1.0, 1.0]
        for i, segment in enumerate(self.stream):
            result = acoustic_richness_index(segment)

            self.assertAlmostEqual(result, expected_results[i], places=2)

    def test_acoustic_entropy_returns_expected_result(self):
        expected_results = [0.891, 0.722, 0.891]
        for i, segment in enumerate(self.stream):
            result = acoustic_entropy(segment)

            self.assertAlmostEqual(result, expected_results[i], places=2)
