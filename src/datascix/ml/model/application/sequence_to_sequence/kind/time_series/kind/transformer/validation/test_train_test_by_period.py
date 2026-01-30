import numpy as np

from datascix.ml.model.application.sequence_to_sequence.kind.time_series.kind.transformer.architecture.architecture import \
    Architecture
from datascix.ml.model.application.sequence_to_sequence.kind.time_series.kind.transformer.predictor.predictor import \
    Predictor
from datascix.ml.model.application.sequence_to_sequence.kind.time_series.kind.transformer.trainer.config import \
    Config as TrainerConfig
from datascix.ml.model.application.sequence_to_sequence.kind.time_series.kind.transformer.trainer.trainer import \
    Trainer
from datascix.ml.model.application.sequence_to_sequence.validation.kind.train_test.train_test_by_periods import \
    TrainTestByPeriods
from mathx.statistic.population.sampling.sampler.kind.countable.finite.members_mentioned.numbered.sequence.sliding_window.sliding_window import \
    SlidingWindow
from utilix.data.storage.kind.file.numpi.multi_valued import MultiValued as NpMultiValued
from utilix.os.file_system.file.file import File as OsFile
from utilix.os.file_system.path.file import File as FilePath


class TestTrainTestByPeriodSampling:
    def test_plot_mean_euclidean_distance_plot(self):
        file_path = FilePath(
            "/home/donkarlo/Dropbox/phd/data/experiements/oldest/robots/uav1/structure/mind/memory/explicit/long_term/episodic/normal/gaussianed_quaternion_kinematic_sliced_from_1_to_300000/time_position_sequence_sliced_from_1_to_300000.npz"
        )
        os_file = OsFile.init_from_path(file_path)
        storage = NpMultiValued(os_file, False)
        storage.load()
        # removing the time
        one_period_members_count = 24450
        ram = storage.get_ram()[:4 * 24450, 1:]  # (T, 3)

        usable_len = (len(ram) // one_period_members_count) * one_period_members_count
        ram = ram[:usable_len]

        partition_count = len(ram) // one_period_members_count

        # shape = (partition_count, one_period_members_count, 3)
        partitioned_population = ram.reshape(partition_count, one_period_members_count, ram.shape[1])
        print(partitioned_population[0].shape)
        three_partitions_concated = np.vstack(
            [partitioned_population[0], partitioned_population[1], partitioned_population[2]])

        print("ram.shape:", ram.shape, "ram.nbytes(MB):", ram.nbytes / 1024 / 1024)
        print("partitioned_population.shape:", partitioned_population.shape, "dtype:", partitioned_population.dtype)

        # cloud_point_math_view = PointCloud(PointGroup(ram))
        # cloud_point_math_view.render()

        # sliding window config
        sliding_window = SlidingWindow(100, 100, 10)

        # architecture config
        feature_dimension = ram.shape[1]
        architecture = Architecture(
            model_dimension=64,
            number_of_attention_heads=8,
            feed_forward_dimension=128,
            input_feature_count=feature_dimension,  # GPS without time has 3 dimensions
            output_time_steps=sliding_window.get_input_length(),  # the length of each sliding window as the prediction length
            output_feature_count=feature_dimension,  # we want 3d GPS predictions
            maximum_time_steps=2048,
            dropout_rate=0.1,
        )

        # training config
        trainer_config = TrainerConfig(
            epochs=10,
            batch_size=8,
            learning_rate=1e-3,
            shuffle=True)

        train_test = TrainTestByPeriods.init_from_one_split(three_partitions_concated, partitioned_population[3],
                                                            Trainer, architecture, trainer_config, Predictor,
                                                            sliding_window.get_step())
        train_test.render_euclidean_distance()
