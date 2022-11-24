import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

import time
from src.analysis.coordinator import AnalysisCoordinator
from src.tools.loader import load_audio

if __name__ == "__main__":
    indices = ["Hf", "Ht", "ACI"]
    audio_stream = load_audio("sample_audio.wav")
    print("Loaded 2:00 audio file")
    num_segments = audio_stream.getNumberOfSegments()
    print("Number of segments: " + str(num_segments))

    startTime = time.time()
    coordinator = AnalysisCoordinator(audio_stream, indices)
    result = coordinator.calculateIndices()
    endTime = time.time()
    print(f"Indices took {endTime - startTime} s to calculate")
    for index in indices:
        print(f"{index}: {result.getResult(index)}")
