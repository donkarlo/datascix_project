import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as TfLayers, Model as TfModel

from datascix.ml.model.application.sequence_to_sequence.time_series_forcating.kind.transformer.architecture.architecture import Architecture
from datascix.ml.model.application.sequence_to_sequence.time_series_forcating.kind.transformer.training.learned_parameters import \
    LearnedParameters


class Forcaster:
    def __init__(
            self,
            architecture: Architecture,
            learned_parameters: LearnedParameters | None = None,
    ):
        self._architecture = architecture
        self._learned_parameters = learned_parameters

        self._output_time_steps = self._architecture.get_output_time_steps()
        self._output_feature_count = self._architecture.get_output_feature_count()

        # Option A: input_feature_count is part of Architecture
        self._input_feature_count = int(self._architecture.get_input_feature_count())

        self._tf_model = self._build_tf_model()

        # build variables once (so set_weights works)
        dummy_b = 1
        dummy_t_in = 1
        dummy_encoder = tf.zeros((dummy_b, dummy_t_in, self._input_feature_count), dtype=tf.float32)
        dummy_decoder = tf.zeros((dummy_b, 1, self._output_feature_count), dtype=tf.float32)
        _ = self._tf_model([dummy_encoder, dummy_decoder], training=False)

        self._maybe_load_weights()

    # -------------------------------------------------
    # INTERNALS
    # -------------------------------------------------

    def _build_tf_model(self) -> TfModel:
        arch = self._architecture

        d_model = arch.get_model_dimension()
        h = arch.get_number_of_attention_heads()
        ff = arch.get_feed_forward_dimension()
        out_f = arch.get_output_feature_count()
        max_steps = arch.get_maximum_time_steps()
        dropout = arch.get_dropout_rate()

        if d_model % h != 0:
            raise ValueError("model_dimension must be divisible by number_of_attention_heads.")
        per_head = d_model // h

        f_in = self._input_feature_count

        encoder_in = TfLayers.Input(shape=(None, f_in), dtype=tf.float32, name="encoder_input")
        decoder_in = TfLayers.Input(shape=(None, out_f), dtype=tf.float32, name="decoder_input")

        enc_proj = TfLayers.Dense(d_model, name="enc_proj")(encoder_in)
        dec_proj = TfLayers.Dense(d_model, name="dec_proj")(decoder_in)

        enc_pos_emb = TfLayers.Embedding(max_steps, d_model, name="enc_pos_emb")
        dec_pos_emb = TfLayers.Embedding(max_steps, d_model, name="dec_pos_emb")

        def add_positional(x, pos_layer):
            t = tf.shape(x)[1]
            pos = pos_layer(tf.range(t))  # (T, D)
            pos = tf.expand_dims(pos, axis=0)  # (1, T, D)
            return x + pos

        enc_x = TfLayers.Lambda(lambda z: add_positional(z, enc_pos_emb), name="enc_add_pos")(enc_proj)
        dec_y = TfLayers.Lambda(lambda z: add_positional(z, dec_pos_emb), name="dec_add_pos")(dec_proj)

        # ENCODER
        enc_attn = TfLayers.MultiHeadAttention(h, per_head, dropout=dropout, name="enc_mha")(
            query=enc_x, value=enc_x, key=enc_x
        )
        enc_x = TfLayers.LayerNormalization(epsilon=1e-6, name="enc_ln1")(enc_x + enc_attn)

        enc_ff = tf.keras.Sequential(
            [TfLayers.Dense(ff, activation="relu"), TfLayers.Dense(d_model)],
            name="enc_ff",
        )(enc_x)
        memory = TfLayers.LayerNormalization(epsilon=1e-6, name="enc_ln2")(enc_x + enc_ff)

        # DECODER
        dec_self = TfLayers.MultiHeadAttention(h, per_head, dropout=dropout, name="dec_mha")(
            query=dec_y, value=dec_y, key=dec_y, use_causal_mask=True
        )
        dec_y = TfLayers.LayerNormalization(epsilon=1e-6, name="dec_ln1")(dec_y + dec_self)

        dec_cross = TfLayers.MultiHeadAttention(h, per_head, dropout=dropout, name="dec_cross")(
            query=dec_y, value=memory, key=memory
        )
        dec_y = TfLayers.LayerNormalization(epsilon=1e-6, name="dec_ln2")(dec_y + dec_cross)

        dec_ff = tf.keras.Sequential(
            [TfLayers.Dense(ff, activation="relu"), TfLayers.Dense(d_model)],
            name="dec_ff",
        )(dec_y)
        dec_y = TfLayers.LayerNormalization(epsilon=1e-6, name="dec_ln3")(dec_y + dec_ff)

        out = TfLayers.Dense(out_f, name="out_proj")(dec_y)

        return TfModel(inputs=[encoder_in, decoder_in], outputs=out, name="transformer_trainable")

    def _maybe_load_weights(self) -> None:
        if self._learned_parameters is None:
            return
        weights = self._learned_parameters.get_weights()
        if weights is None:
            return

        # IMPORTANT: load weights into the INTERNAL tf model (not into Forcaster)
        self._tf_model.set_weights(weights)

    def _build_decoder_input_autoregressive(self, batch_size: tf.Tensor, predicted_so_far: tf.Tensor,
                                            step: int) -> tf.Tensor:
        # predicted_so_far: (B, step, F_out)
        if step == 0:
            return tf.zeros((batch_size, 1, self._output_feature_count), dtype=tf.float32)

        zeros_first = tf.zeros((batch_size, 1, self._output_feature_count), dtype=tf.float32)
        dec_in = tf.concat([zeros_first, predicted_so_far], axis=1)  # (B, step+1, F_out)
        return dec_in

    # -------------------------------------------------
    # PUBLIC API
    # -------------------------------------------------

    def get_forcast(self, time_serie_to_forcast: np.ndarray) -> np.ndarray:
        if not isinstance(time_serie_to_forcast, np.ndarray):
            raise TypeError("time_serie_to_forcast must be np.ndarray.")
        if time_serie_to_forcast.ndim != 3:
            raise ValueError("time_serie_to_forcast must have shape (B, T_in, F_in).")
        if int(time_serie_to_forcast.shape[2]) != self._input_feature_count:
            raise ValueError(
                f"Input feature count must be {self._input_feature_count}, got {int(time_serie_to_forcast.shape[2])}."
            )

        enc = tf.convert_to_tensor(time_serie_to_forcast.astype(np.float32, copy=False), dtype=tf.float32)
        batch_size = tf.shape(enc)[0]

        predicted = tf.zeros((batch_size, 0, self._output_feature_count), dtype=tf.float32)

        for step in range(self._output_time_steps):
            dec_in = self._build_decoder_input_autoregressive(batch_size, predicted, step)
            out = self._tf_model([enc, dec_in], training=False)  # (B, step+1, F_out)
            next_step = out[:, -1:, :]
            predicted = tf.concat([predicted, next_step], axis=1)

        return predicted.numpy()
