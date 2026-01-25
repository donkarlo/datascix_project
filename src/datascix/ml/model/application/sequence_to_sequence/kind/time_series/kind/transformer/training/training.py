import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as TfLayers, Model as TfModel

from datascix.ml.model.application.sequence_to_sequence.kind.time_series.kind.transformer.architecture.architecture import \
    Architecture
from datascix.ml.model.application.sequence_to_sequence.kind.time_series.kind.transformer.training.config import Config
from datascix.ml.model.application.sequence_to_sequence.kind.time_series.kind.transformer.training.learned_parameters import \
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