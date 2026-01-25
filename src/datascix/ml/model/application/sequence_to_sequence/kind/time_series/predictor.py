from abc import abstractmethod, ABC

from datascix.ml.model.application.sequence_to_sequence.predictor import Predictor as SequenceToSequencePredictor
from datascix.ml.model.architecture.architecture import Architecture as ModelArchitecture
import numpy as np

class Predictor(SequenceToSequencePredictor, ABC):
    def __init__(self, achitecture: ModelArchitecture, learned_parameters):
        SequenceToSequencePredictor.__init__(self, achitecture, learned_parameters)

    @abstractmethod
    def get_predictions(self, input_array)->np.ndarray:
        pass