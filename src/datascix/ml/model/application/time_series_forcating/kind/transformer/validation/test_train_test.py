
from datascix.ml.model.application.time_series_forcating.training.sliding_window.generator import Generator
from datascix.ml.model.application.time_series_forcating.training.sliding_window.sliding_window import \
    SlidingWindow
from datascix.ml.model.application.time_series_forcating.kind.transformer.transformer_draft import TransformerDraft

from datascix.ml.model.application.time_series_forcating.validation.kind.train_test import TrainTest
from utilix.data.storage.kind.file.numpi.multi_valued import MultiValued as NpMultiValued
from utilix.os.file_system.file.file import File as OsFile
from utilix.os.file_system.path.file import File as FilePath
from mathx.number.kind.real.interval.unit.open_unit_interval import OpenUnitInterval
from mathx.statistic.population.kind.numpied import Numpied as NumpiedPopulation

class TestTrainTest:
    def test_plot_mean_euclidean_distance_plot(self):
        file_path = FilePath(
            "/home/donkarlo/Dropbox/phd/data/experiements/oldest/robots/uav1/structure/mind/memory/explicit/long_term/episodic/normal/gaussianed_quaternion_kinematic_sliced_from_1_to_300000/time_position_sequence_sliced_from_1_to_300000.npz"
        )
        os_file = OsFile.init_from_path(file_path)
        storage = NpMultiValued(os_file, False)
        storage.load()
        # removing the time
        ram = storage.get_ram()[0:300, 1:]

        sliding_window = SlidingWindow(100, 100, 5)
        sliding_windows_generator = Generator(ram, sliding_window)

        input_array = sliding_windows_generator.get_inputs()
        output_array = sliding_windows_generator.get_outputs()
        input_output_pairs = sliding_windows_generator.get_input_output_pairs()

        transformer_model = TransformerDraft(
            model_dimension=128,
            number_of_attention_heads=8,
            feed_forward_dimension=256,
            output_time_steps=sliding_window.get_output_length(),
            output_feature_count=output_array.shape[2],
            dropout_rate=0.1,
            epochs=2,
            batch_size=5,
        )

        training_population = NumpiedPopulation(input_output_pairs)
        open_unit_interval = OpenUnitInterval(0.7)
        train_test = TrainTest.init_from_partitionaning_ratio(transformer_model, training_population, open_unit_interval)
        train_test.render_euclidean_distance()
