from datascix.ml.model.application.sequence_to_sequence.kind.time_series.kind.auto_regression.vector.Training.config import \
    Config
from datascix.ml.model.application.sequence_to_sequence.kind.time_series.kind.auto_regression.vector.Training.training import \
    Training
from datascix.ml.model.application.sequence_to_sequence.kind.time_series.kind.auto_regression.vector.architecture import \
    Architecture
from datascix.ml.model.application.sequence_to_sequence.kind.time_series.kind.auto_regression.vector.predictor import \
    Predictor
from datascix.ml.model.application.sequence_to_sequence.validation.kind.train_test import TrainTest
from datascix.ml.model.supervision.kind.supervion_dependent.kind.self_supervised.sequence_to_sequence.sliding_window.generator import \
    Generator
from datascix.ml.model.supervision.kind.supervion_dependent.kind.self_supervised.sequence_to_sequence.sliding_window.sliding_window import \
    SlidingWindow
from mathx.number.kind.real.interval.unit.open_unit_interval import OpenUnitInterval
from mathx.statistic.population.kind.numpied import Numpied as NumpiedPopulation
from utilix.data.storage.kind.file.numpi.multi_valued import MultiValued as NpMultiValued
from utilix.os.file_system.file.file import File as OsFile
from utilix.os.file_system.path.file import File as FilePath


class TestTrainTest:

    def test_with_sampling_from_random_sliding_windows_papulation_ratio(self):
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

        architecture = Architecture(
            feature_count=3,
            lag_order=1,
            include_intercept=True
        )

        training_config = Config(fit_method="ols",
                                regularization_strength=None,
                                select_lag_by=None)

        trainer = Training(architecture, training_config, input_target_pairs)
        learned_parameters = trainer.get_learned_parameters()

        predictor = Predictor(architecture, learned_parameters)

        sample_population = NumpiedPopulation(input_target_pairs)
        open_unit_interval = OpenUnitInterval(0.7)
        train_test = TrainTest.init_from_partitionaning_ratio(predictor, sample_population, open_unit_interval)
        train_test.render_euclidean_distance()