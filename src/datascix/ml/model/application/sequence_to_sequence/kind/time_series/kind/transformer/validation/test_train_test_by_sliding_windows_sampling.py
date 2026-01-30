from datascix.ml.model.application.sequence_to_sequence.kind.time_series.kind.transformer.architecture.architecture import \
    Architecture
from datascix.ml.model.application.sequence_to_sequence.kind.time_series.kind.transformer.predictor.predictor import \
    Predictor
from datascix.ml.model.application.sequence_to_sequence.kind.time_series.kind.transformer.trainer.config import Config
from datascix.ml.model.application.sequence_to_sequence.kind.time_series.kind.transformer.trainer.trainer import \
    Trainer
from mathx.statistic.population.sampling.sampler.sampler import Generator
from mathx.statistic.population.sampling.sampler.sampler import SlidingWindow
from datascix.ml.model.application.sequence_to_sequence.validation.kind.train_test.train_test_sliding_window_sampling import \
    TrainTestBySlidingWindowSampling

from mathx.number.kind.real.interval.unit.open_unit_interval import OpenUnitInterval
from mathx.statistic.population.kind.countable.finite.member_mentioned.numbered.numbered import Numbered as NumpiedPopulation
from utilix.data.storage.kind.file.numpi.multi_valued import MultiValued as NpMultiValued
from utilix.os.file_system.file.file import File as OsFile
from utilix.os.file_system.path.file import File as FilePath


class TestTrainTestByPointSampling:
    def test_plot_mean_euclidean_distance_plot(self):
        file_path = FilePath(
            "/home/donkarlo/Dropbox/phd/data/experiements/oldest/robots/uav1/structure/mind/memory/explicit/long_term/episodic/normal/gaussianed_quaternion_kinematic_sliced_from_1_to_300000/time_position_sequence_sliced_from_1_to_300000.npz"
        )
        os_file = OsFile.init_from_path(file_path)
        storage = NpMultiValued(os_file, False)
        storage.load()
        # removing the time
        ram = storage.get_ram()[0:50000, 1:]

        sliding_window = SlidingWindow(100, 100, 5)
        sliding_windows_generator = Generator(ram, sliding_window)

        input_array = sliding_windows_generator.get_inputs()
        target_array = sliding_windows_generator.get_outputs()
        input_target_pairs = sliding_windows_generator.get_input_output_pairs()

        print (f"input_array.shape[2]: {input_array.shape[2]}")
        architecture = Architecture(
            model_dimension=128,
            number_of_attention_heads=8,
            feed_forward_dimension=256,
            input_feature_count=input_array.shape[2],
            output_time_steps=sliding_window.get_output_length(),
            output_feature_count=target_array.shape[2],
            maximum_time_steps=2048,
            dropout_rate=0.1,
        )

        trainer_config = Config(epochs=15,
                                batch_size=16,
                                learning_rate=1e-3,
                                shuffle=True)

        sample_population = NumpiedPopulation(input_target_pairs)
        open_unit_interval = OpenUnitInterval(0.7)
        train_test = TrainTestBySlidingWindowSampling.init_from_partitionaning_ratio(Trainer, architecture, trainer_config, Predictor, sample_population, open_unit_interval)
        train_test.render_euclidean_distance()
