# class for representing audio as a stream of segments

import os

import librosa
import maad

from tools.noise import waveform_denoise, spectrogram_denoise

class AudioSegment:
    def __init__(self, data, sr, denoise = True):
        self.data = data
        self.sr = sr
        self.denoise = denoise
        # caching results
        self._waveform = None
        self._spectrogram = None
    
    def getWaveform(self):
        if self._waveform is not None:
            return self._waveform
        waveform = self.data
        if self.denoise:
            waveform = waveform_denoise(waveform)
        self._waveform = waveform
        return waveform
    
    def getSpectrogram(self):
        if self._spectrogram is not None:
            return self._spectrogram
        spectrogram = maad.sound.spectrogram(self.data, self.sr, window="hamming", nperseg=512)
        if self.denoise:
            spectrogram = spectrogram_denoise(spectrogram)
        self._spectrogram = spectrogram
        return spectrogram
    
class AudioStream(object):
    def __init__(self, file, sr = 22050, segment_length = 60):
        self.file = file
        self.sr = sr
        self.segment_length = segment_length
        # file metadata
        self.duration = librosa.get_duration(file)
        # iteration
        self.position = 0
        
    def __iter__(self):
        return self
    
    def __next__(self):
        return self.next()
        
    def next(self):
        if self.position < self.duration:
            y = librosa.load(self.file, sr=self.sr, offset=self.position, duration=self.segment_length)
            self.position += self.segment_length
            return AudioSegment(y, sr=self.sr)
        raise StopIteration()
    
def load_audio(file):
    _, ext = os.path.splitext(file)
    if ext != ".wav" or not os.path.exists(file):
        return

    return AudioStream(file)