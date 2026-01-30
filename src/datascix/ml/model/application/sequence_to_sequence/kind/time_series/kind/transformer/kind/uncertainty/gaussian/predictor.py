# file: datascix/ml/model/application/sequence_to_sequence/kind/time_series/kind/transformer/kind/uncertainty/gaussian/predictor.py
from __future__ import annotations

import numpy as np
import tensorflow as tf

from datascix.ml.model.application.sequence_to_sequence.kind.time_series.kind.transformer.kind.uncertainty.gaussian.architecture import \
    Architecture
from datascix.ml.model.application.sequence_to_sequence.trainer.learned_parameters import LearnedParameters


class Predictor:
    """Predict Gaussian parameters (mu, var) for each step and feature.

    Notes:
        - The underlying model outputs concatenated parameters: [mu, log_var] along the last axis.
        - var is computed as exp(log_var).
    """

    def __init__(self, architecture: Architecture, learned_parameters: LearnedParameters | None = None):
        self._architecture = architecture
        self._learned_parameters = learned_parameters

        self._output_time_steps = int(self._architecture.get_output_time_steps())
        self._input_feature_count = int(self._architecture.get_input_feature_count())
        self._output_feature_count = int(self._architecture.get_output_feature_count())

        self._tf_model = self._architecture.build_tf_model()
        self._build_once_for_set_weights()
        self._maybe_load_weights()

    def _build_once_for_set_weights(self) -> None:
        dummy_batch_size = 1
        dummy = tf.zeros((dummy_batch_size, self._output_time_steps, self._input_feature_count), dtype=tf.float32)
        _ = self._tf_model(dummy, training=False)

    def _maybe_load_weights(self) -> None:
        if self._learned_parameters is None:
            return

        weights = self._learned_parameters.get_weights()
        if weights is None:
            return

        self._tf_model.set_weights(weights)

    def get_predicted_distributions(self, input_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(input_array, np.ndarray):
            raise TypeError("input_array must be np.ndarray.")
        if input_array.ndim != 3:
            raise ValueError("input_array must have shape (B, T, F).")

        if int(input_array.shape[1]) != self._output_time_steps:
            raise ValueError("input_array time steps mismatch.")
        if int(input_array.shape[2]) != self._input_feature_count:
            raise ValueError("input_array feature count mismatch.")

        x = tf.convert_to_tensor(input_array.astype(np.float32, copy=False), dtype=tf.float32)
        y = self._tf_model(x, training=False)

        mu = y[..., :self._output_feature_count]
        log_var = y[..., self._output_feature_count:]

        var = tf.exp(log_var)

        return mu.numpy(), var.numpy()
