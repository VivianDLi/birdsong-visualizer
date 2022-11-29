import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

import time
from numpy.testing import assert_array_almost_equal

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
    print("Saving to .csv file")
    coordinator.saveIndices("sample_audio.csv")
    print("Loading Indices from .csv file")
    startTime = time.time()
    new_coordinator = AnalysisCoordinator(audio_stream, indices)
    new_result = new_coordinator.loadIndices("sample_audio.csv")
    endTime = time.time()
    print(f"Indices took {endTime - startTime} s to load")
    print("Checking the two calculations are equal...")
    for index in indices:
        assert_array_almost_equal(
            result.getResult(index), new_result.getResult(index)
        )
    print("Done!")
