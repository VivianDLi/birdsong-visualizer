# class for storing result analysis

class Result:
    def __init__(self, result, segment_length):
        self.result = result
        self.segment_length = segment_length

    def getIndexValues(self, index):
        return self.result[index]

    def segmentToTimestamp(self, seg_num):
        return self.segment_length * seg_num
