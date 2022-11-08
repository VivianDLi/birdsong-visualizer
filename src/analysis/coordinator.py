# coordinate multiple analyzers in parallel

class AnalysisCoordinator:
    def __init__(self, stream: AudioStream, indices):
        supported_indices = ["Ht", "M", "BgN", "SNR", "AcAct", "AEFrac", "AEDur", "Hf", "HfVar", "HfMax",
                             "SpDiv", "SpAct", "ACI", "AEI", "BioI", "LFreqCov", "MFreqCov", "HFreqCov", "NDSI", "ARI", "H"]
        if len(indices) > 3:
            raise ValueError(
                "Only up to three acoustic indices should be specified.")
        if any([i in supported_indices for i in indices]):
            raise ValueError("Unsupported acoustic indices were specified.")
        self.stream = stream
        self.indices = indices
        self.analyzers = [Analyzer(segment, indices) for segment in stream]

    def calculateIndices(self):
        pass
