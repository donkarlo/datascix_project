import numpy as np

from datascix.ml.model.application.sequence_to_sequence.predictor import Predictor
from datascix.ml.model.application.sequence_to_sequence.trainer.trainer import Trainer
from datascix.ml.model.architecture.architecture import Architecture
from datascix.ml.model.supervision.kind.supervion_dependent.training.config import Config
from mathx.view.kind.point_cloud.decorator.lined.ordered_intra_line_connected import OrderedIntraLineConnected
from datascix.ml.model.validation.validation import Validation
from mathx.set_nd.decorator.factory.numpied import Numpied as NumpiedSet
from mathx.number.kind.real.interval.unit.open_unit_interval import OpenUnitInterval
from mathx.statistic.population.kind.countable.finite.member_mentioned.numbered.numbered import Numbered as NumpiedPopulation
from mathx.view.kind.point_cloud.kind.multiple_point_group.multiple_point_grouped import MultiplePointGrouped
from mathx.view.kind.point_cloud.decorator.lined.group_point_seted.ordered_inter_line_connected import OrderedInterLineConnected
from mathx.view.kind.point_cloud.point_cloud import PointCloud
from mathx.view.kind.point_cloud.point.group.group.Group import Group as GroupPairSet
from mathx.view.kind.point_cloud.point.group.group import Group


class TrainTestBySlidingWindowSampling(Validation):
    def __init__(self, predictor: Predictor, train_set: NumpiedSet, test_set: NumpiedSet):
        self._test_set_predictions: np.ndarray | None = None
        self._test_set_target_values: np.ndarray | None = None
        self._predictor = predictor

        self._train_set_pairs = np.asarray(train_set.get_members())
        self._train_set_inputs = self._train_set_pairs[:, 0]
        self._train_set_targets = self._train_set_pairs[:, 1]

        self._test_set_pairs = np.asarray(test_set.get_members())
        self._test_set_inputs = self._test_set_pairs[:, 0]
        self._test_set_targets = self._test_set_pairs[:, 1]

        self._test()

    def _test(self) -> None:
        self._test_set_predictions = self._predictor.get_predictions(self._test_set_inputs)

    def render_euclidean_distance(self):
        residuals = self._test_set_predictions - self._test_set_targets
        residuals = np.linalg.norm(residuals, axis=-1)
        distance_curve = residuals.mean(axis=1)

        residuals = distance_curve  # shape: (N,)
        indices = np.arange(1, len(residuals) + 1)
        two_dimensional = np.column_stack((indices, residuals))

        pair_set = Group(two_dimensional)
        point_cloud = OrderedIntraLineConnected(PointCloud(pair_set))
        point_cloud._build()
        point_cloud.render()

    def render_line_connected_corresponding_pairs(self):
        """
        This only works if the data is 2d or 3d
        Returns:

        """
        test_set_targets = Group(self._test_set_targets)
        test_set_predictions = Group(self._test_set_predictions)
        test_set_predictions_group_pair_set = GroupPairSet([test_set_predictions])

        line_connected_multi_data_set = OrderedInterLineConnected(
            MultiplePointGrouped(PointCloud(test_set_targets), test_set_predictions_group_pair_set))
        line_connected_multi_data_set.show()

    @classmethod
    def init_from_partitionaning_ratio(cls, trainer_class: Trainer, architecture: Architecture, trainer_configs: Config,
                                       predictor_class: Predictor, sample_population: NumpiedPopulation,
                                       ratio: OpenUnitInterval) -> None:
        subset_complement_partition = sample_population.get_random_sample_and_complement_by_ratio(ratio)

        train_set = subset_complement_partition.get_subset()
        test_set = subset_complement_partition.get_complement()

        trainer = trainer_class(architecture, trainer_configs, train_set.get_members())
        learned_parameters = trainer.get_learned_parameters()

        predictor = predictor_class(architecture, learned_parameters)

        return cls(predictor, train_set, test_set)