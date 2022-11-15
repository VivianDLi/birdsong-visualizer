# class for storing result analysis

from typing import List


class Result:
    def __init__(self, result, segment_length: float):
        self.result = result
        self.segment_length = segment_length

    def getIndexValues(self, index: int) -> List[float]:
        return self.result[index]

    def segmentToTimestamp(self, seg_num: int) -> float:
        return self.segment_length * seg_num
