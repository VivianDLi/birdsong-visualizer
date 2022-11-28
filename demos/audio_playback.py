import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

import time
from src.tools.loader import load_audio

if __name__ == "__main__":
    audio_stream = load_audio("sample_audio.wav")
    print("Loaded 2:00 audio file")
    print("Playing first 20 seconds")
    audio_stream.play(0, 20)
    time.sleep(20)
    print("Playing from 60 s for a minute")
    audio_stream.play(60, 60)
    time.sleep(5)
    print("Interrupting to play from 30 s for 10 s")
    audio_stream.play(30, 10)
    time.sleep(5)
    print("Stopping abruptly after 5 s")
    audio_stream.stop()
