# class for representing audio as a stream of segments

import os
from typing import List, Tuple, Union
import threading
import queue

import librosa
import maad
import numpy as np
import sounddevice as sd

from src.tools.noise import waveform_denoise, spectrogram_denoise
from src.tools.interfaces import IAudioSegment, IAudioStream


class PlaybackThread(threading.Thread):
    def __init__(
        self, stream, sr: int, block_size: int, buffer_size: int = 10
    ):
        threading.Thread.__init__(self, daemon=True)
        self.stream = stream
        self.sr = sr
        self.block_size = block_size
        self.buffer_size = buffer_size
        self.queue = queue.Queue(maxsize=buffer_size)
        self.event = threading.Event()
        self.output_stream = None

    def _callback(self, outdata, frames, time, status):
        assert frames == self.block_size
        if status.output_underflow:
            raise sd.CallbackAbort("Output underflow: increase blocksize?")
        assert not status
        try:
            data = self.queue.get_nowait()
            data = np.reshape(data, (len(data), 1))
        except queue.Empty:
            raise sd.CallbackAbort("Buffer is empty: increase buffersize?")
        if len(data) < len(outdata):
            outdata[: len(data)] = data
            outdata[len(data) :].fill(0)
            raise sd.CallbackStop
        else:
            outdata[:] = data

    def run(self):
        for y_block in (
            x for _, x in zip(range(self.buffer_size), self.stream)
        ):
            self.queue.put_nowait(y_block)
        self.output_stream = sd.OutputStream(
            samplerate=self.sr,
            channels=1,
            blocksize=self.block_size,
            callback=self._callback,
            finished_callback=self.event.set,
        )
        with self.output_stream:
            for y_block in self.stream:
                self.queue.put(y_block, block=True, timeout=None)
            self.event.wait()

    def stop(self):
        if self.output_stream is not None:
            self.output_stream.abort()
        raise KeyboardInterrupt()


class AudioSegment(IAudioSegment):
    def __init__(self, data, sr: int, denoise: bool = True):
        self.data = data
        self.sr = sr
        self.denoise = denoise
        # caching results
        self._waveform = None
        self._spectrogram, self._tn, self._fn = None, None, None
        self._bg_noise = None

    # in dB using average as baseline dB
    def getWaveform(self) -> np.ndarray:
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
    ) -> Tuple[np.ndarray, List[float], List[float]]:
        if (
            self._spectrogram is not None
            and self._tn is not None
            and self._fn is not None
        ):
            return self._spectrogram, self._tn, self._fn
        spectrogram, tn, fn, _ = maad.sound.spectrogram(
            self.data, self.sr, window="hamming", mode="psd", nperseg=512
        )
        if self.denoise:
            spectrogram = spectrogram_denoise(spectrogram)
        self._spectrogram, self._tn, self._fn = spectrogram, tn, fn
        return spectrogram, tn, fn


class AudioStream(IAudioStream):
    def __init__(
        self,
        file: str,
        sr: int = 22050,
        duration: float = 60,
        time_limits: Union[Tuple[float, float], None] = None,
    ):
        self.file = file
        self.sr = sr
        self.segment_duration = duration
        # file metadata
        self.file_duration: float = librosa.get_duration(filename=file)
        if time_limits is None:
            self.time_limits = (0, self.file_duration)
        else:
            self.time_limits = time_limits
        # iteration
        self.position = 0
        # playback
        self._playback_thread = None

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.position < self.file_duration:
            y, sr = librosa.load(
                self.file,
                sr=self.sr,
                offset=self.position,
                duration=self.segment_duration,
            )
            self.position += self.segment_duration
            return AudioSegment(y, sr=sr)
        raise StopIteration()

    def createStream(
        self,
        start_time: float,
        end_time: float,
        segment_duration: Union[float, None] = None,
    ) -> IAudioStream:
        if segment_duration is None:
            segment_duration = (
                end_time - start_time
            ) / self.getNumberOfSegments()
        return AudioStream(
            self.file, self.sr, segment_duration, (start_time, end_time)
        )

    def createSTFT(
        self, n_fft: int = 2048, hop_length: int = 1024
    ) -> np.ndarray:
        stream = librosa.stream(
            self.file,
            block_length=256,
            frame_length=n_fft,
            hop_length=hop_length,
            offset=self.time_limits[0],
            duration=self.time_limits[1] - self.time_limits[0],
        )
        # concatenate block stfts horizontally
        S = np.concatenate(
            [
                librosa.stft(
                    y_block, n_fft=n_fft, hop_length=hop_length, center=False
                )
                for y_block in stream
            ],
            axis=1,
        )
        # and return the amplitude
        return np.abs(S)

    def play(
        self, offset: float = 0, duration: Union[float, None] = None
    ) -> None:
        if self._playback_thread is not None:
            self.stop()
        if duration is None:
            duration = self.file_duration - offset
        stream = librosa.stream(
            self.file,
            block_length=10,
            frame_length=1024,
            hop_length=1024,
            offset=offset,
            duration=duration,
        )
        self._playback_thread = PlaybackThread(stream, self.sr, 10 * 1024)
        self._playback_thread.start()

    def stop(self):
        if self._playback_thread is not None:
            try:
                self._playback_thread.stop()
            except KeyboardInterrupt:
                self._playback_thread = None


def load_audio(file: str) -> IAudioStream:
    _, ext = os.path.splitext(file)
    if ext.lower() != ".wav":
        raise ValueError("file is not a .wav")
    if not os.path.exists(file):
        raise ValueError("file doesn't exist")

    return AudioStream(file)
