from datascix.ml.kind.dimension_reduction.pca.decorator.decorator import Decorator as BaseDecorator
from datascix.ml.kind.sequence_to_sequence.kind.transformer.interface import Interface


class Decorator(BaseDecorator):
    def __init__(self, inner: Interface):
        BaseDecorator.__init__(self, inner)