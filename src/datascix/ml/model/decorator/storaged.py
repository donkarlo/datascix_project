from datascix.ml.model.application.time_series_forcating.decorator.decorator import Decorator
from datascix.ml.model.application.time_series_forcating.kind.transformer.interface import Interface
from utilix.data.storage.interface import Interface as StorageInterface

class Storaged(Decorator):
    def __init__(self, inner:Interface):
        pass