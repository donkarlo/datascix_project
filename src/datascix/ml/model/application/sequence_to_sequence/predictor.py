from datascix.ml.model.application.sequence_to_sequence.trainer.learned_parameters import LearnedParameters
from datascix.ml.model.architecture.architecture import Architecture as ModelArchitecture


class Predictor:
    def __init__(self, achitecture: ModelArchitecture, learned_parameters:LearnedParameters):
        self._architecture = achitecture
        self._learned_parameters = learned_parameters

    def get_learned_parameters(self)-> LearnedParameters:
        return self._learned_parameters

    def get_architecture(self) -> ModelArchitecture:
        return self._architecture