man
mikham
kode
transformer
ro
baraye
dadeye
se
bodiye
GPS
bedam
ke
ye
drone
dore
ye
mostatil
charkhodeh.mikham
khoob
negah
koni
bebini
chizi
ezafe
ya
na
marboot
hast
ya
na

from datascix.ml.model.application.sequence_to_sequence.kind.time_series.kind.transformer.predictor import \
    Predictor
from datascix.ml.model.application.sequence_to_sequence.kind.time_series.kind.transformer.trainer.trainer import \
    Trainer
from datascix.ml.model.supervision.kind.supervion_dependent.kind.self_supervised.sequence_to_sequence.sliding_window.generator import \
    Generator
from datascix.ml.model.supervision.kind.supervion_dependent.kind.self_supervised.sequence_to_sequence.sliding_window.sliding_window import \
    SlidingWindow
from datascix.ml.model.application.sequence_to_sequence.validation.kind.train_test.train_test_sliding_window_sampling import \
    TrainTestBySlidingWindowSampling

from utilix.data.storage.kind.file.numpi.multi_valued import MultiValued as NpMultiValued
from utilix.os.file_system.file.file import File as OsFile
from utilix.os.file_system.path.file import File as FilePath


class TestTrainTest:
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

        trainer_config = Config(epochs=10,
                                batch_size=16,
                                learning_rate=1e-3,
                                shuffle=True)

        trainer = Trainer(architecture, trainer_config, input_target_pairs)
        learned_parameters = trainer.get_learned_parameters()

        predictor = Predictor(architecture, learned_parameters)

        sample_population = NumpiedPopulation(input_target_pairs)
        open_unit_interval = OpenUnitInterval(0.7)
        train_test = TrainTestBySlidingWindowSampling.init_from_partitionaning_ratio(predictor, sample_population, open_unit_interval)
        train_test.render_euclidean_distance()


class Architecture:
    def __init__(
            self,
            model_dimension: int,
            number_of_attention_heads: int,
            feed_forward_dimension: int,
            input_feature_count: int,
            output_time_steps: int,
            output_feature_count: int,
            maximum_time_steps: int = 2048,
            dropout_rate: float = 0.1,
    ):

        self._model_dimension = int(model_dimension)

        self._number_of_attention_heads = int(number_of_attention_heads)
        self._feed_forward_dimension = int(feed_forward_dimension)
        self._input_feature_count = int(input_feature_count)
        self._output_time_steps = int(output_time_steps)
        self._output_feature_count = int(output_feature_count)
        self._maximum_time_steps = int(maximum_time_steps)
        self._dropout_rate = float(dropout_rate)

        if self._input_feature_count <= 0:
            raise ValueError("input_feature_count must be > 0.")
        if self._model_dimension <= 0:
            raise ValueError("model_dimension must be > 0.")
        if self._number_of_attention_heads <= 0:
            raise ValueError("number_of_attention_heads must be > 0.")
        if self._feed_forward_dimension <= 0:
            raise ValueError("feed_forward_dimension must be > 0.")
        if self._output_time_steps <= 0:
            raise ValueError("output_time_steps must be > 0.")
        if self._output_feature_count <= 0:
            raise ValueError("output_feature_count must be > 0.")
        if self._maximum_time_steps <= 0:
            raise ValueError("maximum_time_steps must be > 0.")
        if self._model_dimension % self._number_of_attention_heads != 0:
            raise ValueError("model_dimension must be divisible by number_of_attention_heads.")
        if self._dropout_rate < 0.0 or self._dropout_rate >= 1.0:
            raise ValueError("dropout_rate must be in [0.0, 1.0).")

    def get_model_dimension(self) -> int:
        return self._model_dimension

    def get_number_of_attention_heads(self) -> int:
        return self._number_of_attention_heads

    def get_feed_forward_dimension(self) -> int:
        return self._feed_forward_dimension

    def get_input_feature_count(self) -> int:
        return self._input_feature_count

    def get_output_time_steps(self) -> int:
        return self._output_time_steps

    def get_output_feature_count(self) -> int:
        return self._output_feature_count

    def get_maximum_time_steps(self) -> int:
        return self._maximum_time_steps

    def get_dropout_rate(self) -> float:
        return self._dropout_rate


class Config:
    def __init__(
            self,
            epochs: int,
            batch_size: int,
            learning_rate: float = 1e-3,
            shuffle: bool = True,
    ):

        self._epochs = int(epochs)

        self._batch_size = int(batch_size)
        self._learning_rate = float(learning_rate)
        self._shuffle = bool(shuffle)

        if self._epochs <= 0:
            raise ValueError("epochs must be > 0.")
        if self._batch_size <= 0:
            raise ValueError("batch_size must be > 0.")
        if self._learning_rate <= 0.0:
            raise ValueError("learning_rate must be > 0.")

    def get_epochs(self) -> int:
        return self._epochs

    def get_batch_size(self) -> int:
        return self._batch_size

    def get_learning_rate(self) -> float:
        return self._learning_rate

    def get_shuffle(self) -> bool:
        return self._shuffle


import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as TfLayers, Model as TfModel

from datascix.ml.model.application.sequence_to_sequence.kind.time_series.kind.transformer.architecture.architecture import \
    Architecture
from datascix.ml.model.application.sequence_to_sequence.kind.time_series.kind.transformer.trainer.config import Config
from datascix.ml.model.application.sequence_to_sequence.kind.time_series.kind.transformer.trainer.learned_parameters import \
    LearnedParameters


class Training:
    """Train a seq2seq Transformer and expose only learned weights.

    This implementation is intentionally aligned with TransformerDraft:
        - Learned positional embeddings
        - 1 encoder block: MHA -> Dropout -> Residual -> LayerNorm -> FFN -> Dropout -> Residual -> LayerNorm
        - 1 decoder block: causal MHA -> Dropout -> Residual -> LayerNorm -> cross MHA -> Dropout -> Residual -> LayerNorm
                          -> FFN -> Dropout -> Residual -> LayerNorm -> output projection
        - Teacher forcing decoder input: [0, y_0, ..., y_{T-2}]

    Notes:
        - No normalization is applied here.
        - The public API is restricted to get_architecture() and get_learned_parameters().
    """

    def __init__(
            self,
            architecture: Architecture,
            config: Config,
            input_target_pairs: np.ndarray
    ):
        self._architecture = architecture
        self._config = config

        input_array = input_target_pairs[:, 0]
        target_array = input_target_pairs[:, 1]

        if not isinstance(input_array, np.ndarray) or not isinstance(target_array, np.ndarray):
            raise TypeError("input_array and target_array must be np.ndarray.")
        if input_array.ndim != 3 or target_array.ndim != 3:
            raise ValueError("input_array and target_array must have shape (B, T, F).")
        if input_array.shape[0] != target_array.shape[0]:
            raise ValueError("Batch size mismatch between input_array and target_array.")

        f_in_expected = int(self._architecture.get_input_feature_count())
        f_in_got = int(input_array.shape[2])
        if f_in_got != f_in_expected:
            raise ValueError(f"input_array feature count must be {f_in_expected}, got {f_in_got}.")

        t_out_expected = int(self._architecture.get_output_time_steps())
        f_out_expected = int(self._architecture.get_output_feature_count())

        if int(target_array.shape[1]) != t_out_expected:
            raise ValueError(f"target_array time steps must be {t_out_expected}, got {int(target_array.shape[1])}.")
        if int(target_array.shape[2]) != f_out_expected:
            raise ValueError(f"target_array feature count must be {f_out_expected}, got {int(target_array.shape[2])}.")

        self._input_array = input_array.astype(np.float32, copy=False)
        self._target_array = target_array.astype(np.float32, copy=False)

        self._learned_parameters: LearnedParameters | None = None
        self._has_trained = False

        self._tf_model = self._build_tf_model()
        self._train_once()

    def _build_tf_model(self) -> TfModel:
        arch = self._architecture

        d_model = arch.get_model_dimension()
        heads = arch.get_number_of_attention_heads()
        ff_dim = arch.get_feed_forward_dimension()

        f_in = arch.get_input_feature_count()
        f_out = arch.get_output_feature_count()

        max_steps = arch.get_maximum_time_steps()
        dropout_rate = arch.get_dropout_rate()

        if d_model % heads != 0:
            raise ValueError("model_dimension must be divisible by number_of_attention_heads.")
        per_head = d_model // heads

        encoder_in = TfLayers.Input(shape=(None, f_in), dtype=tf.float32, name="encoder_input")
        decoder_in = TfLayers.Input(shape=(None, f_out), dtype=tf.float32, name="decoder_input")

        # Projections
        enc_proj = TfLayers.Dense(d_model, name="enc_proj")(encoder_in)
        dec_proj = TfLayers.Dense(d_model, name="dec_proj")(decoder_in)

        # Learned positional embeddings
        enc_pos_emb = TfLayers.Embedding(max_steps, d_model, name="enc_pos_emb")
        dec_pos_emb = TfLayers.Embedding(max_steps, d_model, name="dec_pos_emb")

        def add_positional(x, pos_layer):
            time_steps = tf.shape(x)[1]
            positions = pos_layer(tf.range(time_steps))
            positions = tf.expand_dims(positions, axis=0)
            return x + positions

        enc_x = TfLayers.Lambda(lambda z: add_positional(z, enc_pos_emb), name="enc_add_pos")(enc_proj)
        dec_y = TfLayers.Lambda(lambda z: add_positional(z, dec_pos_emb), name="dec_add_pos")(dec_proj)

        # ----------------------
        # ENCODER (1 block) aligned with TransformerDraft
        # ----------------------
        enc_mha_layer = TfLayers.MultiHeadAttention(
            num_heads=heads,
            key_dim=per_head,
            dropout=dropout_rate,
            name="enc_mha",
        )
        enc_attn = enc_mha_layer(query=enc_x, value=enc_x, key=enc_x)
        enc_attn = TfLayers.Dropout(dropout_rate, name="enc_dropout_after_attention")(enc_attn)
        enc_x = TfLayers.LayerNormalization(epsilon=1e-6, name="enc_ln_after_attention")(enc_x + enc_attn)

        enc_ff = tf.keras.Sequential(
            [TfLayers.Dense(ff_dim, activation="relu"), TfLayers.Dense(d_model)],
            name="enc_ff",
        )(enc_x)
        enc_ff = TfLayers.Dropout(dropout_rate, name="enc_dropout_after_ff")(enc_ff)
        memory = TfLayers.LayerNormalization(epsilon=1e-6, name="enc_ln_after_ff")(enc_x + enc_ff)

        # ----------------------
        # DECODER (1 block) aligned with TransformerDraft
        # ----------------------
        dec_self_mha_layer = TfLayers.MultiHeadAttention(
            num_heads=heads,
            key_dim=per_head,
            dropout=dropout_rate,
            name="dec_self_mha",
        )
        dec_self = dec_self_mha_layer(query=dec_y, value=dec_y, key=dec_y, use_causal_mask=True)
        dec_self = TfLayers.Dropout(dropout_rate, name="dec_dropout_after_self_attention")(dec_self)
        dec_y = TfLayers.LayerNormalization(epsilon=1e-6, name="dec_ln_after_self")(dec_y + dec_self)

        dec_cross_mha_layer = TfLayers.MultiHeadAttention(
            num_heads=heads,
            key_dim=per_head,
            dropout=dropout_rate,
            name="dec_cross_mha",
        )
        dec_cross = dec_cross_mha_layer(query=dec_y, value=memory, key=memory)
        dec_cross = TfLayers.Dropout(dropout_rate, name="dec_dropout_after_cross_attention")(dec_cross)
        dec_y = TfLayers.LayerNormalization(epsilon=1e-6, name="dec_ln_after_cross")(dec_y + dec_cross)

        dec_ff = tf.keras.Sequential(
            [TfLayers.Dense(ff_dim, activation="relu"), TfLayers.Dense(d_model)],
            name="dec_ff",
        )(dec_y)
        dec_ff = TfLayers.Dropout(dropout_rate, name="dec_dropout_after_ff")(dec_ff)
        dec_y = TfLayers.LayerNormalization(epsilon=1e-6, name="dec_ln_after_ff")(dec_y + dec_ff)

        out = TfLayers.Dense(f_out, name="out_proj")(dec_y)

        return TfModel(inputs=[encoder_in, decoder_in], outputs=out, name="transformer_trainable")

    def _build_teacher_forcing_decoder_input(self, target_tensor: tf.Tensor) -> tf.Tensor:
        batch_size = tf.shape(target_tensor)[0]
        f_out = self._architecture.get_output_feature_count()
        zeros_first = tf.zeros((batch_size, 1, f_out), dtype=target_tensor.dtype)
        shifted = target_tensor[:, :-1, :]
        return tf.concat([zeros_first, shifted], axis=1)

    def _train_once(self) -> None:
        if self._has_trained:
            return

        enc = tf.convert_to_tensor(self._input_array, dtype=tf.float32)
        tgt = tf.convert_to_tensor(self._target_array, dtype=tf.float32)
        dec = self._build_teacher_forcing_decoder_input(tgt)

        # Ensure variables are created
        _ = self._tf_model([enc[:1], dec[:1]], training=False)

        self._tf_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self._config.get_learning_rate()),
            loss=tf.keras.losses.MeanSquaredError(),
        )

        self._tf_model.fit(
            [enc, dec],
            tgt,
            epochs=self._config.get_epochs(),
            batch_size=self._config.get_batch_size(),
            shuffle=self._config.get_shuffle(),
        )

        self._learned_parameters = LearnedParameters(weights=self._tf_model.get_weights())
        self._has_trained = True

    def get_architecture(self) -> Architecture:
        return self._architecture

    def get_learned_parameters(self) -> LearnedParameters:
        if self._learned_parameters is None:
            raise RuntimeError("learned_parameters is not available.")
        return self._learned_parameters


import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as TfLayers, Model as TfModel

from datascix.ml.model.application.sequence_to_sequence.kind.time_series.kind.transformer.architecture.architecture import \
    Architecture
from datascix.ml.model.application.sequence_to_sequence.trainer.learned_parameters import LearnedParameters


class Predictor:
    """Autoregressive forecaster aligned with TransformerDraft.

    This class builds the same network as Trainer and then loads weights via set_weights.
    Inference is strictly autoregressive and starts with an all-zero first token.
    """

    def __init__(
            self,
            architecture: Architecture,
            learned_parameters: LearnedParameters | None = None,
    ):
        self._architecture = architecture
        self._learned_parameters = learned_parameters

        self._output_time_steps = self._architecture.get_output_time_steps()
        self._output_feature_count = self._architecture.get_output_feature_count()
        self._input_feature_count = self._architecture.get_input_feature_count()

        self._tf_model = self._build_tf_model()

        # Build variables once (so set_weights is valid)
        dummy_b = 1
        dummy_t_in = 1
        dummy_encoder = tf.zeros((dummy_b, dummy_t_in, self._input_feature_count), dtype=tf.float32)
        dummy_decoder = tf.zeros((dummy_b, 1, self._output_feature_count), dtype=tf.float32)
        _ = self._tf_model([dummy_encoder, dummy_decoder], training=False)

        self._maybe_load_weights()

    def _build_tf_model(self) -> TfModel:
        arch = self._architecture

        d_model = arch.get_model_dimension()
        heads = arch.get_number_of_attention_heads()
        ff_dim = arch.get_feed_forward_dimension()

        f_in = arch.get_input_feature_count()
        f_out = arch.get_output_feature_count()

        max_steps = arch.get_maximum_time_steps()
        dropout_rate = arch.get_dropout_rate()

        if d_model % heads != 0:
            raise ValueError("model_dimension must be divisible by number_of_attention_heads.")
        per_head = d_model // heads

        encoder_in = TfLayers.Input(shape=(None, f_in), dtype=tf.float32, name="encoder_input")
        decoder_in = TfLayers.Input(shape=(None, f_out), dtype=tf.float32, name="decoder_input")

        enc_proj = TfLayers.Dense(d_model, name="enc_proj")(encoder_in)
        dec_proj = TfLayers.Dense(d_model, name="dec_proj")(decoder_in)

        enc_pos_emb = TfLayers.Embedding(max_steps, d_model, name="enc_pos_emb")
        dec_pos_emb = TfLayers.Embedding(max_steps, d_model, name="dec_pos_emb")

        def add_positional(x, pos_layer):
            time_steps = tf.shape(x)[1]
            positions = pos_layer(tf.range(time_steps))
            positions = tf.expand_dims(positions, axis=0)
            return x + positions

        enc_x = TfLayers.Lambda(lambda z: add_positional(z, enc_pos_emb), name="enc_add_pos")(enc_proj)
        dec_y = TfLayers.Lambda(lambda z: add_positional(z, dec_pos_emb), name="dec_add_pos")(dec_proj)

        # ENCODER (aligned with Trainer / TransformerDraft)
        enc_mha_layer = TfLayers.MultiHeadAttention(
            num_heads=heads,
            key_dim=per_head,
            dropout=dropout_rate,
            name="enc_mha",
        )
        enc_attn = enc_mha_layer(query=enc_x, value=enc_x, key=enc_x)
        enc_attn = TfLayers.Dropout(dropout_rate, name="enc_dropout_after_attention")(enc_attn)
        enc_x = TfLayers.LayerNormalization(epsilon=1e-6, name="enc_ln_after_attention")(enc_x + enc_attn)

        enc_ff = tf.keras.Sequential(
            [TfLayers.Dense(ff_dim, activation="relu"), TfLayers.Dense(d_model)],
            name="enc_ff",
        )(enc_x)
        enc_ff = TfLayers.Dropout(dropout_rate, name="enc_dropout_after_ff")(enc_ff)
        memory = TfLayers.LayerNormalization(epsilon=1e-6, name="enc_ln_after_ff")(enc_x + enc_ff)

        # DECODER (aligned)
        dec_self_mha_layer = TfLayers.MultiHeadAttention(
            num_heads=heads,
            key_dim=per_head,
            dropout=dropout_rate,
            name="dec_self_mha",
        )
        dec_self = dec_self_mha_layer(query=dec_y, value=dec_y, key=dec_y, use_causal_mask=True)
        dec_self = TfLayers.Dropout(dropout_rate, name="dec_dropout_after_self_attention")(dec_self)
        dec_y = TfLayers.LayerNormalization(epsilon=1e-6, name="dec_ln_after_self")(dec_y + dec_self)

        dec_cross_mha_layer = TfLayers.MultiHeadAttention(
            num_heads=heads,
            key_dim=per_head,
            dropout=dropout_rate,
            name="dec_cross_mha",
        )
        dec_cross = dec_cross_mha_layer(query=dec_y, value=memory, key=memory)
        dec_cross = TfLayers.Dropout(dropout_rate, name="dec_dropout_after_cross_attention")(dec_cross)
        dec_y = TfLayers.LayerNormalization(epsilon=1e-6, name="dec_ln_after_cross")(dec_y + dec_cross)

        dec_ff = tf.keras.Sequential(
            [TfLayers.Dense(ff_dim, activation="relu"), TfLayers.Dense(d_model)],
            name="dec_ff",
        )(dec_y)
        dec_ff = TfLayers.Dropout(dropout_rate, name="dec_dropout_after_ff")(dec_ff)
        dec_y = TfLayers.LayerNormalization(epsilon=1e-6, name="dec_ln_after_ff")(dec_y + dec_ff)

        out = TfLayers.Dense(f_out, name="out_proj")(dec_y)

        return TfModel(inputs=[encoder_in, decoder_in], outputs=out, name="transformer_trainable")

    def _maybe_load_weights(self) -> None:
        if self._learned_parameters is None:
            return
        weights = self._learned_parameters.get_weights()
        if weights is None:
            return
        self._tf_model.set_weights(weights)

    def _build_decoder_input_autoregressive(self, batch_size: tf.Tensor, predicted_so_far: tf.Tensor,
                                            step: int) -> tf.Tensor:
        if step == 0:
            return tf.zeros((batch_size, 1, self._output_feature_count), dtype=tf.float32)

        zeros_first = tf.zeros((batch_size, 1, self._output_feature_count), dtype=tf.float32)
        return tf.concat([zeros_first, predicted_so_far], axis=1)

    def get_predictions(self, time_serie_to_forcast: np.ndarray) -> np.ndarray:
        if not isinstance(time_serie_to_forcast, np.ndarray):
            raise TypeError("time_serie_to_forcast must be np.ndarray.")
        if time_serie_to_forcast.ndim != 3:
            raise ValueError("time_serie_to_forcast must have shape (B, T_in, F_in).")
        if int(time_serie_to_forcast.shape[2]) != int(self._input_feature_count):
            raise ValueError(
                f"Input feature count must be {int(self._input_feature_count)}, got {int(time_serie_to_forcast.shape[2])}."
            )

        enc = tf.convert_to_tensor(time_serie_to_forcast.astype(np.float32, copy=False), dtype=tf.float32)
        batch_size = tf.shape(enc)[0]

        predicted = tf.zeros((batch_size, 0, self._output_feature_count), dtype=tf.float32)

        for step in range(int(self._output_time_steps)):
            dec_in = self._build_decoder_input_autoregressive(batch_size, predicted, step)
            out = self._tf_model([enc, dec_in], training=False)
            next_step = out[:, -1:, :]
            predicted = tf.concat([predicted, next_step], axis=1)

        return predicted.numpy()


import numpy as np

from datascix.ml.model.application.sequence_to_sequence.predictor import Predictor
from mathx.view.kind.point_cloud.decorator.lined.ordered_intra_line_connected import OrderedIntraLineConnected
from datascix.ml.model.validation.validation import Validation
from mathx.set_nd.decorator.factory.numpied import Numpied as NumpiedSet
from mathx.number.kind.real.interval.unit.open_unit_interval import OpenUnitInterval
from mathx.statistic.population.kind.countable.finite.member_mentioned.numbered import Numbered as NumpiedPopulation
from mathx.view.kind.point_cloud.kind.multiple_point_group.multiple_point_grouped import MultiplePointGrouped
from mathx.view.kind.point_cloud.decorator.lined.group_point_seted.ordered_inter_line_connected import \
    OrderedInterLineConnected
from mathx.view.kind.point_cloud.point_cloud import PointCloud
from mathx.view.kind.point_cloud.point.group.group.Group import Group as GroupPairSet
from mathx.view.kind.point_cloud.point.group.group import Group


class TrainTest(Validation):
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
    def init_from_partitionaning_ratio(cls, forcaster: Predictor, sample_population: NumpiedPopulation,
                                       ratio: OpenUnitInterval) -> None:
        subset_complement_partition = sample_population.get_random_sample_and_complement_by_ratio(ratio)
        train_set = subset_complement_partition.get_subset()
        test_set = subset_complement_partition.get_complement()
        return cls(forcaster, train_set, test_set)
