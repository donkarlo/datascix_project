import numpy as np

from datascix.ml.kind.sequence_to_sequence.kind.transformer.transformer import Transformer
from mathx.setex.decorator.factory.numpied import Numpied as NumpiedSet
from mathx.number.kind.real.interval.unit.open_unit_interval import OpenUnitInterval
from mathx.statistic.population.kind.numpied import Numpied as NumpiedPopulation


class TrainTest:
    def __init__(self, model:Transformer, train_set: NumpiedSet, test_set: NumpiedSet):
        self._test_set_predictions:np.ndarray = None
        self._test_set_target_values:np.ndarray = None

        self._model = model

        self._train_set_pairs = train_set
        self._train_set_inputs = self._train_set_pairs[:0]
        self._train_set_targets = self._train_set_pairs[:1]

        self._test_set_pairs = test_set
        self._test_set_inputs = self._test_set_pairs[:0]
        self._test_set_targets = self._test_set_pairs[:1]

        self._train()
        self._test()


    def _train(self)->None:
        self._model.fit(self._train_set_inputs, self._train_set_targets)

    def _test(self)->None:
        self._test_set_predictions = self._model.predict(self._test_set_inputs, self._test_set_targets)

    def plot_euclidean_distance(self):
        pass

    def plot_line_connected_corresponding_pairs(self):
        data_sets = DataSets[DataSet(self._test_set_targets), DataSet(self._test_set_predictions)]
        line_connected_multi_data_set = LineConnected[MultiDataSet(Scatter(), data_sets)]
        line_connected_multi_data_set.show()

    @classmethod
    def init_from_partitionaning_ratio(cls, model:Transformer, population: NumpiedPopulation, ratio: OpenUnitInterval)->None:
        subset_complement_partition = population.get_random_sample_and_complement_by_ratio(ratio)
        train_set = subset_complement_partition.get_subset()
        test_set = subset_complement_partition.get_complement()
        return cls(model, train_set, test_set)