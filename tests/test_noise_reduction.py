import operator
import unittest

import numpy as np
from numpy.testing import assert_array_compare

from src.tools.loader import load_audio, AudioSegment


class TestNoiseReduction(unittest.TestCase):
    def assertArrayLessThanOrEqual(self, a, b, msg=None):
        try:
            assert_array_compare(
                operator.__le__, a, b, verbose=True, equal_inf=False
            )
        except AssertionError:
            raise self.failureException(msg)

    def setUp(self):
        self.stream = load_audio("./tests/test.wav")

        self.addTypeEqualityFunc(np.ndarray, self.assertArrayLessThanOrEqual)

    def test_acoustic_noise_reduction_returns_same_size_array(self):
        for segment in self.stream:
            original_audio = segment.data
            denoised_audio = segment.getWaveform()

            self.assertEqual(len(denoised_audio), len(original_audio))

    def test_acoustic_noise_reduction_removes_noise(self):
        for segment in self.stream:
            original_audio = segment.data
            unnoised_audio = AudioSegment(
                original_audio, segment.sr, denoise=False
            ).getWaveform()
            denoised_audio = segment.getWaveform()
            noise = segment.getNoise()

            self.assertAlmostEqual(noise, -49.45)
            self.assertEqual(unnoised_audio, denoised_audio)

    def test_spectral_noise_reduction_returns_same_size_array(self):
        for segment in self.stream:
            spectrogram, _, _ = segment.getSpectrogram()

            self.assertEqual(
                spectrogram.shape,
                (256, 5166),
            )

    def test_spectral_noise_reduction_removes_noise(self):
        for segment in self.stream:
            original_audio = segment.data
            unnoised_spectrogram, _, _ = AudioSegment(
                original_audio, segment.sr, denoise=False
            ).getSpectrogram()
            denoised_spectrogram, _, _ = segment.getSpectrogram()

            self.assertEqual(denoised_spectrogram, unnoised_spectrogram)
