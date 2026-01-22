from abc import ABC, abstractmethod

from datascix.ml.model.application.time_series_forcating.parameter.parameters import Parameters as TimeSeriesForcastingParameters
from datascix.ml.model.model import Model
import numpy as np

class TimeSeriesForcasting(Model, ABC):

    def __init__(self, parameters: TimeSeriesForcastingParameters):
        self._parameters = parameters

    @abstractmethod
    def get_forcast(self)->np.ndarray:
        pass