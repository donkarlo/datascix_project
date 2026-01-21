from abc import ABC, abstractmethod

from datascix.ml.model.model import Model
import numpy as np

class TimeSeriesForcasting(Model, ABC):
    @abstractmethod
    def get_forcast(self)->np.ndarray:
        pass