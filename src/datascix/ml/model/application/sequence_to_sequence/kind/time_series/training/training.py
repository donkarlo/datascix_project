from abc import abstractmethod
from datascix.ml.model.architecture.architecture import Architecture
from datascix.ml.model.architecture.config import Config
from datascix.ml.model.supervision.kind.supervion_dependent.training.training import Training as SupervionDependentTraining
from datascix.ml.model.supervision.kind.supervion_dependent.training.learned_parameters import LearnedParameters as SupervisionDependentLearnedParameters
import numpy as np

class Training(SupervionDependentTraining):
    def __init__(self, architecture: Architecture, config:Config, input_traget_pairs:np.ndarray):
        SupervionDependentTraining.__init__(self, architecture, config, input_traget_pairs)
        self._learned_parameters = None

    @abstractmethod
    def get_learned_parameters(self)-> SupervisionDependentLearnedParameters:
        ...