from datascix.ml.model.application.sequence_to_sequence.trainer.learned_parameters import LearnedParameters
from datascix.ml.model.architecture.architecture import Architecture


class Trainer:
    def __init__(self, architecture: Architecture, learned_parameters):
        self._architecture = architecture
        self._learned_parameters = learned_parameters

    def get_architecture(self) -> Architecture:
        return self._architecture

    def get_learned_parameters(self) -> LearnedParameters:
        return self._learned_parameters
