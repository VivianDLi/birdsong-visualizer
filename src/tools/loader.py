# class for representing audio as a stream of segments

import os
from typing import List, Tuple, Any, Union

import librosa
import maad
import numpy as np
import sounddevice as sd

from src.tools.noise import waveform_denoise, spectrogram_denoise


class AudioSegment:
    def __init__(
        self, data, sr: int, denoise: bool = True, loop: bool = False
    ):
        self.data = data
        self.sr = sr
        self.denoise = denoise
        # caching results
        self._waveform = None
        self._spectrogram, self._tn, self._fn = None, None, None
        self._bg_noise = None

        # audio playback
        self.loop = loop
        self._is_playing = False

    # in dB using average as baseline dB
    def getWaveform(self) -> List[float]:
        if self._waveform is not None:
            return self._waveform
        # convert to dB based on average amplitude
        avg = np.average(self.data)
        waveform = 20 * np.log10(np.abs(self.data / avg))
        if self.denoise:
            waveform, noise = waveform_denoise(waveform)
            self._bg_noise = noise
        self._waveform = waveform
        return waveform

    def getNoise(self) -> float:
        if self._bg_noise is not None:
            return self._bg_noise
        avg = np.average(self.data)
        waveform = 20 * np.log10(np.abs(self.data / avg))
        _, noise = waveform_denoise(waveform)
        self._bg_noise = noise
        return noise

    def getSpectrogram(
        self,
    ) -> Tuple[List[List[float]], List[float], List[float]]:
        if (
            self._spectrogram is not None
            and self._tn is not None
            and self._fn is not None
        ):
            return self._spectrogram, self._tn, self._fn
        spectrogram: List[List[float]]
        tn: List[float]
        fn: List[float]
        spectrogram, tn, fn, _ = maad.sound.spectrogram(
            self.data, self.sr, window="hamming", mode="psd", nperseg=512
        )
        if self.denoise:
            spectrogram = spectrogram_denoise(spectrogram)
        self._spectrogram, self._tn, self._fn = spectrogram, tn, fn
        return spectrogram, tn, fn

    def play(self):
        sd.play(self.data, self.sr, loop=self.loop)
        self._is_playing = True

    def stop(self):
        if self._is_playing:
            sd.stop()
            self._is_playing = False


class AudioStream(object):
    def __init__(self, file: str, sr: int = 22050, segment_length: float = 60):
        self.file = file
        self.sr = sr
        self.segment_length = segment_length
        # file metadata
        self.duration: float = librosa.get_duration(filename=file)
        # iteration
        self.position = 0

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.position < self.duration:
            y, sr = librosa.load(
                self.file,
                sr=self.sr,
                offset=self.position,
                duration=self.segment_length,
            )
            self.position += self.segment_length
            return AudioSegment(y, sr=sr)
        raise StopIteration()

    def getNumSegments(self) -> int:
        # ceiling division through negation
        return int(-(self.duration // -self.segment_length))

    def segmentToTimestamp(self, seg_num: int) -> float:
        return self.segment_length * seg_num

    def getSegment(
        self,
        start_time: float,
        end_time: Union[float, None] = None,
        duration: Union[float, None] = None,
        loop: bool = False,
    ) -> AudioSegment:
        if end_time is None and duration is None:
            raise ValueError("One of end_time or duration must be specified")
        if end_time is not None:
            duration = end_time - start_time
        y, sr = librosa.load(
            self.file, sr=self.sr, offset=start_time, duration=duration
        )
        return AudioSegment(y, sr=sr, loop=loop)


def load_audio(file: str):
    _, ext = os.path.splitext(file)
    if ext != ".wav":
        raise ValueError("file is not a .wav")
    if not os.path.exists(file):
        raise ValueError("file doesn't exist")

    return AudioStream(file)
