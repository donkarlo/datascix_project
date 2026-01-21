from abc import ABC, abstractmethod
from typing import Protocol


class Interface(ABC, Protocol):
    @abstractmethod
    def predict(self, sequence_to_predict:np.nd_array)->np.nd_array:
        pass