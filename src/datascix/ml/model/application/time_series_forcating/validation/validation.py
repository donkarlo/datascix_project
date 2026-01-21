import numpy as np

from datascix.ml.model.application.time_series_forcating.kind.transformer.transformer import Transformer
from mathx.number.kind.real.interval.unit.open_unit_interval import OpenUnitInterval
from mathx.statistic.population.kind.numpied import Numpied as NumpiedPopulation


class Validation:
    def __init__(self, model:Transformer, inputs:np.ndarray, targets:np.ndarray, predictions: np.ndarray):
        pass

    def init_from_partitionaning_ratio(self, population: NumpiedPopulation, ratio:OpenUnitInterval):
        pass




