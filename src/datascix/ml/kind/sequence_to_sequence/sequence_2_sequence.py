from abc import ABC, abstractmethod

import numpy as np


class Sequence2Sequence(ABC):

    @abstractmethod
    def predict(self, input:np.ndarray)->np.ndarray:
        pass