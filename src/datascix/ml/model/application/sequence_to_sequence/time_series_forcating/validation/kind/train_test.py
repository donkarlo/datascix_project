import numpy as np

from datascix.ml.model.application.sequence_to_sequence.time_series_forcating.kind.transformer.forcaster import Forcaster
from mathx.view.kind.point_cloud.decorator.lined.ordered_intra_line_connected import OrderedIntraLineConnected
from datascix.ml.model.validation.validation import Validation
from mathx.setex.decorator.factory.numpied import Numpied as NumpiedSet
from mathx.number.kind.real.interval.unit.open_unit_interval import OpenUnitInterval
from mathx.statistic.population.kind.numpied import Numpied as NumpiedPopulation
from mathx.view.kind.point_cloud.decorator.lined.group_pair_seted.group_pair_seted import GroupPairSeted
from mathx.view.kind.point_cloud.decorator.lined.group_pair_seted.ordered_inter_line_connected import OrderedInterLineConnected
from mathx.view.kind.point_cloud.point_cloud import PointCloud
from mathx.view.pair_set.group.Group import Group as GroupPairSet
from mathx.view.pair_set.pair_set import PairSet


class TrainTest(Validation):
    def __init__(self, forcaster:Forcaster, train_set: NumpiedSet, test_set: NumpiedSet):
        self._test_set_predictions: np.ndarray | None = None
        self._test_set_target_values: np.ndarray | None = None
        self._forcaster = forcaster

        self._train_set_pairs = np.asarray(train_set.get_members())
        self._train_set_inputs = self._train_set_pairs[:, 0]
        self._train_set_targets = self._train_set_pairs[:, 1]

        self._test_set_pairs = np.asarray(test_set.get_members())
        self._test_set_inputs = self._test_set_pairs[:, 0]
        self._test_set_targets = self._test_set_pairs[:, 1]

        self._test()

    def _test(self)->None:
        self._test_set_predictions = self._forcaster.get_forcast(self._test_set_inputs)



    def render_euclidean_distance(self):
        differences = self._test_set_predictions - self._test_set_targets
        distances = np.linalg.norm(differences, axis=-1)
        distance_curve = distances.mean(axis=1)

        distances = distance_curve  # shape: (N,)
        indices = np.arange(1, len(distances) + 1)
        two_dimensional = np.column_stack((indices, distances))

        pair_set = PairSet(two_dimensional)
        point_cloud = OrderedIntraLineConnected(PointCloud(pair_set))
        point_cloud.build()
        point_cloud.render()

    def render_line_connected_corresponding_pairs(self):
        """
        This only works if the data is 2d or 3d
        Returns:

        """
        test_set_targets = PairSet(self._test_set_targets)
        test_set_predictions = PairSet(self._test_set_predictions)
        test_set_predictions_group_pair_set = GroupPairSet([test_set_predictions])

        line_connected_multi_data_set = OrderedInterLineConnected(GroupPairSeted(PointCloud(test_set_targets), test_set_predictions_group_pair_set))
        line_connected_multi_data_set.show()

    @classmethod
    def init_from_partitionaning_ratio(cls, forcaster: Forcaster, population: NumpiedPopulation, ratio: OpenUnitInterval)->None:
        subset_complement_partition = population.get_random_sample_and_complement_by_ratio(ratio)
        train_set = subset_complement_partition.get_subset()
        test_set = subset_complement_partition.get_complement()
        return cls(forcaster, train_set, test_set)