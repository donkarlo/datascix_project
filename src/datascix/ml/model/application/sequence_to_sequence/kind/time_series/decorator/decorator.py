from datascix.ml.model.application.dimension_reduction.pca.decorator.decorator import Decorator as BaseDecorator
from datascix.ml.model.application.sequence_to_sequence.kind.time_series_forcating.kind.transformer.interface import Interface


class Decorator(BaseDecorator):
    def __init__(self, inner: Interface):
        BaseDecorator.__init__(self, inner)