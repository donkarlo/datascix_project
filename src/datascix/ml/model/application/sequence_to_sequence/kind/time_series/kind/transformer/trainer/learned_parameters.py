from typing import List, Optional
class LearnedParameters:
    def __init__(self, weights: Optional[List] = None):
        self._weights = weights


    def get_weights(self):
        return self._weights
