# trainer.py
from __future__ import annotations

import numpy as np
import tensorflow as tf

from datascix.ml.model.application.time_series_forcating.kind.transformer.architecture.architecture import Architecture
from datascix.ml.model.application.time_series_forcating.kind.transformer.training.config import Config
from datascix.ml.model.application.time_series_forcating.kind.transformer.training.learned_parameters import \
    LearnedParameters
from datascix.ml.model.application.time_series_forcating.kind.transformer.forcaster import Forcaster


class Trainer:
    """
    Hidden training orchestration.

    Construction triggers training once.

    Public API (ONLY):
        - get_architecture()
        - get_learned_parameters()
    """

    def __init__(
            self,
            architecture: Architecture,
            config: Config,
            input_array: np.ndarray,
            target_array: np.ndarray,
    ):
        self._architecture = architecture
        self._config = config

        if not isinstance(input_array, np.ndarray) or not isinstance(target_array, np.ndarray):
            raise TypeError("input_array and target_array must be np.ndarray.")
        if input_array.ndim != 3 or target_array.ndim != 3:
            raise ValueError("input_array and target_array must have shape (B, T, F).")
        if input_array.shape[0] != target_array.shape[0]:
            raise ValueError("Batch size mismatch between input_array and target_array.")
        if target_array.shape[1] != architecture.get_output_time_steps():
            raise ValueError(
                f"target_array time steps must be {architecture.get_output_time_steps()}, got {target_array.shape[1]}."
            )
        if target_array.shape[2] != architecture.get_output_feature_count():
            raise ValueError(
                f"target_array feature count must be {architecture.get_output_feature_count()}, got {target_array.shape[2]}."
            )

        self._input_array = input_array.astype(np.float32, copy=False)
        self._target_array = target_array.astype(np.float32, copy=False)

        self._learned_parameters: LearnedParameters | None = None
        self._has_trained = False

        self._train_once()

    # -------------------------------------------------
    # INTERNALS
    # -------------------------------------------------

    def _build_teacher_forcing_decoder_input(self, target_tensor: tf.Tensor) -> tf.Tensor:
        b = tf.shape(target_tensor)[0]
        f = self._architecture.get_output_feature_count()
        zeros = tf.zeros((b, 1, f), dtype=target_tensor.dtype)
        shifted = target_tensor[:, :-1, :]
        return tf.concat([zeros, shifted], axis=1)

    def _train_once(self) -> None:
        if self._has_trained:
            return

        # 1) Build a trainable model with the SAME layer structure as inference Forcaster
        model = Forcaster(architecture=self._architecture, learned_parameters=None)

        enc = tf.convert_to_tensor(self._input_array, dtype=tf.float32)
        tgt = tf.convert_to_tensor(self._target_array, dtype=tf.float32)
        dec = self._build_teacher_forcing_decoder_input(tgt)

        # 2) Build variables (Dense kernels need known feature dim)
        _ = model((enc[:1], dec[:1]), training=False)

        # 3) Compile & fit
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self._config.get_learning_rate()),
            loss=tf.keras.losses.MeanSquaredError(),
        )
        model.fit(
            (enc, dec),
            tgt,
            epochs=self._config.get_epochs(),
            batch_size=self._config.get_batch_size(),
            shuffle=self._config.get_shuffle(),
        )

        # 4) Store learned things: weights + input feature count
        self._learned_parameters = LearnedParameters(
            model_weights=model.get_weights(),
            input_feature_count=int(self._input_array.shape[2]),
        )
        self._has_trained = True

    # -------------------------------------------------
    # ONLY PUBLIC API
    # -------------------------------------------------

    def get_architecture(self) -> Architecture:
        return self._architecture

    def get_learned_parameters(self) -> LearnedParameters:
        if self._learned_parameters is None:
            raise RuntimeError("learned_parameters is not available.")
        return self._learned_parameters
