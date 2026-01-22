import numpy as np
import tensorflow as tf
from tensorflow.keras import Model as TfModel

from datascix.ml.model.application.time_series_forcating.kind.transformer.config.config import Config
from datascix.ml.model.application.time_series_forcating.kind.transformer.training.learned_parameters import \
    LearnedParameters


class Training(TfModel):
    """
    Hidden training orchestration.

    Public API:
        - get_architecture()
        - get_learned_parameters()
    """

    def __init__(
            self,
            *,
            architecture,
            draft_model: TfModel,
            config:Config,
            input_array: np.ndarray,
            target_array: np.ndarray,
    ):
        TfModel.__init__(self)

        self._architecture = architecture
        self._draft_model = draft_model
        self._config = config

        self._input_array = input_array.astype(np.float32, copy=False)
        self._target_array = target_array.astype(np.float32, copy=False)

        self._learned_parameters = None
        self._has_trained = False

    # -------------------------------------------------
    # INTERNALS
    # -------------------------------------------------

    def _compute_learned_parameters(self):
        x = self._input_array
        y = self._target_array

        input_mean = x.mean(axis=(0, 1))
        input_std = x.std(axis=(0, 1))
        target_mean = y.mean(axis=(0, 1))
        target_std = y.std(axis=(0, 1))

        eps = np.float32(1e-8)
        input_std = np.maximum(input_std, eps)
        target_std = np.maximum(target_std, eps)

        return LearnedParameters(
            input_mean=input_mean,
            input_std=input_std,
            target_mean=target_mean,
            target_std=target_std,
        )

    def _normalize_arrays(self, learned):
        x = (self._input_array - learned.get_input_mean()) / learned.get_input_std()
        y = (self._target_array - learned.get_target_mean()) / learned.get_target_std()
        return x, y

    def _build_teacher_forcing_decoder_input(self, target_tensor):
        output_features = self._architecture.get_output_feature_count()
        b = tf.shape(target_tensor)[0]
        zeros = tf.zeros((b, 1, output_features), dtype=target_tensor.dtype)
        shifted = target_tensor[:, :-1, :]
        return tf.concat([zeros, shifted], axis=1)

    def _train_once(self):
        if self._has_trained:
            return

        learned = self._compute_learned_parameters()
        x_norm, y_norm = self._normalize_arrays(learned)

        encoder_input = tf.convert_to_tensor(x_norm)
        target_tensor = tf.convert_to_tensor(y_norm)
        decoder_input = self._build_teacher_forcing_decoder_input(target_tensor)

        # build weights
        _ = self._draft_model((encoder_input[:1], decoder_input[:1]), training=False)

        self._draft_model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self._config.get_learning_rate()
            ),
            loss=tf.keras.losses.MeanSquaredError(),
        )

        self._draft_model.fit(
            (encoder_input, decoder_input),
            target_tensor,
            epochs=self._config.get_epochs(),
            batch_size=self._config.get_batch_size(),
            shuffle=self._config.get_shuffle(),
        )

        self._learned_parameters = learned
        self._has_trained = True

    # -------------------------------------------------
    # ONLY PUBLIC API
    # -------------------------------------------------

    def get_architecture(self):
        return self._architecture

    def get_learned_parameters(self):
        self._train_once()
        return self._learned_parameters
