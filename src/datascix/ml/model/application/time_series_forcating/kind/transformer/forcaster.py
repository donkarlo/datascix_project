import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as TfLayers, Model as TfModel

from datascix.ml.model.application.time_series_forcating.kind.transformer.architecture.architecture import Architecture
from datascix.ml.model.application.time_series_forcating.kind.transformer.training.learned_parameters import \
    LearnedParameters


class Forcaster(TfModel):
    def __init__(
            self,
            architecture: Architecture,
            learned_parameters: LearnedParameters | None = None,
    ):
        TfModel.__init__(self)

        self._architecture = architecture
        self._learned_parameters = learned_parameters

        model_dimension = self._architecture.get_model_dimension()
        num_heads = self._architecture.get_number_of_attention_heads()
        ff_dim = self._architecture.get_feed_forward_dimension()
        self._output_time_steps = self._architecture.get_output_time_steps()
        self._output_feature_count = self._architecture.get_output_feature_count()
        max_steps = self._architecture.get_maximum_time_steps()
        dropout = self._architecture.get_dropout_rate()

        per_head_dim = model_dimension // num_heads

        # IMPORTANT: keep layer structure compatible with Draft
        self.encoder_input_projection = TfLayers.Dense(model_dimension)
        self.decoder_input_projection = TfLayers.Dense(model_dimension)
        self.output_projection = TfLayers.Dense(self._output_feature_count)

        self.encoder_position_embedding = TfLayers.Embedding(max_steps, model_dimension)
        self.decoder_position_embedding = TfLayers.Embedding(max_steps, model_dimension)

        self.encoder_self_attention = TfLayers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=per_head_dim,
            dropout=dropout,
        )
        self.encoder_feed_forward = tf.keras.Sequential(
            [
                TfLayers.Dense(ff_dim, activation="relu"),
                TfLayers.Dense(model_dimension),
            ]
        )
        self.encoder_layer_norm_after_attention = TfLayers.LayerNormalization(epsilon=1e-6)
        self.encoder_layer_norm_after_feed_forward = TfLayers.LayerNormalization(epsilon=1e-6)

        self.decoder_self_attention = TfLayers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=per_head_dim,
            dropout=dropout,
        )
        self.decoder_cross_attention = TfLayers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=per_head_dim,
            dropout=dropout,
        )
        self.decoder_feed_forward = tf.keras.Sequential(
            [
                TfLayers.Dense(ff_dim, activation="relu"),
                TfLayers.Dense(model_dimension),
            ]
        )
        self.decoder_layer_norm_after_self_attention = TfLayers.LayerNormalization(epsilon=1e-6)
        self.decoder_layer_norm_after_cross_attention = TfLayers.LayerNormalization(epsilon=1e-6)
        self.decoder_layer_norm_after_feed_forward = TfLayers.LayerNormalization(epsilon=1e-6)

    # ----- normalization: safe by default -----
    def _normalize_input_np(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)

        if self._learned_parameters is None or not self._learned_parameters.has_input_normalization():
            return x

        mean = self._learned_parameters.get_input_mean()
        std = self._learned_parameters.get_input_std()

        if mean.shape != (x.shape[2],) or std.shape != (x.shape[2],):
            raise ValueError(f"Input normalization shape mismatch: mean/std must be ({x.shape[2]},)")

        eps = np.float32(1e-8)
        std = np.maximum(std, eps)
        return (x - mean) / std

    def _denormalize_target_np(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float32)

        if self._learned_parameters is None or not self._learned_parameters.has_target_normalization():
            return y

        mean = self._learned_parameters.get_target_mean()
        std = self._learned_parameters.get_target_std()

        if mean.shape != (y.shape[2],) or std.shape != (y.shape[2],):
            raise ValueError(f"Target normalization shape mismatch: mean/std must be ({y.shape[2]},)")

        eps = np.float32(1e-8)
        std = np.maximum(std, eps)
        return y * std + mean

    # ----- core -----
    def _add_positional_embedding(self, projected: tf.Tensor, embedding: TfLayers.Embedding) -> tf.Tensor:
        t = tf.shape(projected)[1]
        max_steps = tf.cast(embedding.input_dim, t.dtype)
        tf.debugging.assert_less_equal(t, max_steps, message="Sequence length exceeds positional embedding capacity.")
        pos = tf.range(t)
        pos = embedding(pos)
        pos = tf.expand_dims(pos, axis=0)
        return projected + pos

    def _encode(self, encoder_input: tf.Tensor, training: bool) -> tf.Tensor:
        x = self.encoder_input_projection(encoder_input)
        x = self._add_positional_embedding(x, self.encoder_position_embedding)

        attn = self.encoder_self_attention(query=x, value=x, key=x, training=training)
        x = self.encoder_layer_norm_after_attention(x + attn)

        ff = self.encoder_feed_forward(x, training=training)
        x = self.encoder_layer_norm_after_feed_forward(x + ff)
        return x

    def _decode(self, memory: tf.Tensor, decoder_input: tf.Tensor, training: bool) -> tf.Tensor:
        y = self.decoder_input_projection(decoder_input)
        y = self._add_positional_embedding(y, self.decoder_position_embedding)

        self_attn = self.decoder_self_attention(
            query=y, value=y, key=y, use_causal_mask=True, training=training
        )
        y = self.decoder_layer_norm_after_self_attention(y + self_attn)

        cross = self.decoder_cross_attention(query=y, value=memory, key=memory, training=training)
        y = self.decoder_layer_norm_after_cross_attention(y + cross)

        ff = self.decoder_feed_forward(y, training=training)
        y = self.decoder_layer_norm_after_feed_forward(y + ff)

        return self.output_projection(y)

    def call(self, inputs, training: bool = False) -> tf.Tensor:
        encoder_input, decoder_input = inputs
        memory = self._encode(encoder_input, training=training)
        return self._decode(memory, decoder_input, training=training)

    def get_forcast(self, time_serie_to_forcast: np.ndarray) -> np.ndarray:
        if not isinstance(time_serie_to_forcast, np.ndarray):
            raise TypeError("time_serie_to_forcast must be np.ndarray.")
        if time_serie_to_forcast.ndim != 3:
            raise ValueError("time_serie_to_forcast must have shape (B, T_in, F_in).")

        x = self._normalize_input_np(time_serie_to_forcast)
        encoder_input = tf.convert_to_tensor(x, dtype=tf.float32)
        memory = self._encode(encoder_input, training=False)

        batch_size = tf.shape(encoder_input)[0]
        seq = tf.zeros((batch_size, 0, self._output_feature_count), dtype=tf.float32)

        for step in range(self._output_time_steps):
            if step == 0:
                dec_in = tf.zeros((batch_size, 1, self._output_feature_count), dtype=tf.float32)
            else:
                dec_in = tf.concat(
                    [tf.zeros((batch_size, 1, self._output_feature_count), dtype=tf.float32), seq],
                    axis=1,
                )
                dec_in = dec_in[:, : step + 1, :]

            out = self._decode(memory, dec_in, training=False)
            next_step = out[:, -1:, :]
            seq = tf.concat([seq, next_step], axis=1)

        y = seq.numpy()
        y = self._denormalize_target_np(y)
        return y