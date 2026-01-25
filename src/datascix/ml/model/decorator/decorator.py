from datascix.ml.model.application.dimension_reduction.pca.decorator.decorator import Decorator as BaseDecorator
from datascix.ml.model.application.sequence_to_sequence.kind.time_series_forcating.kind.transformer.transformer_draft import TransformerDraft


class Decorator(BaseDecorator):
    def __init__(self, inner: TransformerDraft):
        BaseDecorator.__init__(self, inner)