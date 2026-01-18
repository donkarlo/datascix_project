from abc import ABC, abstractmethod

import numpy as np
from datascix.ml.model.model import Model
from mathx.statistic.population.kind.finite_numbers_defined_by_members import FiniteNumbersDefinedByMembers


class Sequence2Sequence(Model, ABC):

    def __init__(self):
        self._is_trained = False
        self_training_set = None

    @abstractmethod
    def predict(self, input:np.ndarray)->np.ndarray:
        pass