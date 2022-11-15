import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

import time
import sounddevice as sd
from src.tools.loader import load_audio

if __name__ == "__main__":
    audio_stream = load_audio("sample_audio.wav")
    print("Loaded 59:55 audio file")
    first_20 = audio_stream.getSegment(0, duration=20)
    print("Playing first 20 seconds")
    first_20.play()
    sd.wait()
    middle_20 = audio_stream.getSegment(60, 80, loop=True)
    print("Playing loop from 60 s for a minute")
    middle_20.play()
    time.sleep(60)
    middle_20.stop()
