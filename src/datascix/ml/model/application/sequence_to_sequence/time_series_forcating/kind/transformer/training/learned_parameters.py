import numpy as np


class LearnedParameters:
    def __init__(self, weights: list[np.ndarray]):
        self._weights = [np.asarray(w) for w in weights]

    def get_weights(self) -> list[np.ndarray]:
        return self._weights
