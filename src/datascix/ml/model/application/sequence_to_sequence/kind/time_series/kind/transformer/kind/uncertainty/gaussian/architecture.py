# file: datascix/ml/model/application/sequence_to_sequence/kind/time_series/kind/transformer/kind/uncertainty/gaussian/architecture.py
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers as TfLayers, Model as TfModel

from datascix.ml.model.application.sequence_to_sequence.kind.time_series.kind.transformer.architecture.architecture import \
    Architecture as BaseTransformerArchitecture


class Architecture(BaseTransformerArchitecture):
    """Vanilla direct seq2seq Transformer that outputs concatenated Gaussian parameters.

    Output tensor shape:
        - y: (B, T_out, 2 * F_out)
            first  F_out: mu
            second F_out: log_var (diagonal, log-space)
    """

    def build_tf_model(self) -> TfModel:
        model_dimension = int(self.get_model_dimension())
        number_of_attention_heads = int(self.get_number_of_attention_heads())
        feed_forward_dimension = int(self.get_feed_forward_dimension())
        input_feature_count = int(self.get_input_feature_count())
        output_time_steps = int(self.get_output_time_steps())
        output_feature_count = int(self.get_output_feature_count())
        maximum_time_steps = int(self.get_maximum_time_steps())
        dropout_rate = float(self.get_dropout_rate())

        if model_dimension % number_of_attention_heads != 0:
            raise ValueError("model_dimension must be divisible by number_of_attention_heads.")
        key_dimension = model_dimension // number_of_attention_heads

        x_in = TfLayers.Input(shape=(output_time_steps, input_feature_count), dtype=tf.float32, name="x_in")

        x = TfLayers.Dense(model_dimension, name="in_proj")(x_in)

        position_embedding = TfLayers.Embedding(maximum_time_steps, model_dimension, name="pos_emb")

        def add_positional(tensor: tf.Tensor) -> tf.Tensor:
            time_steps = tf.shape(tensor)[1]
            positions = position_embedding(tf.range(time_steps))
            positions = tf.expand_dims(positions, axis=0)
            return tensor + positions

        x = TfLayers.Lambda(add_positional, name="add_pos")(x)

        attention_layer = TfLayers.MultiHeadAttention(
            num_heads=number_of_attention_heads,
            key_dim=key_dimension,
            dropout=dropout_rate,
            name="mha",
        )
        attention = attention_layer(query=x, value=x, key=x)
        attention = TfLayers.Dropout(dropout_rate, name="drop_after_attn")(attention)
        x = TfLayers.LayerNormalization(epsilon=1e-6, name="ln_after_attn")(x + attention)

        feed_forward = tf.keras.Sequential(
            [TfLayers.Dense(feed_forward_dimension, activation="relu"), TfLayers.Dense(model_dimension)],
            name="ffn",
        )(x)
        feed_forward = TfLayers.Dropout(dropout_rate, name="drop_after_ffn")(feed_forward)
        x = TfLayers.LayerNormalization(epsilon=1e-6, name="ln_after_ffn")(x + feed_forward)

        mu = TfLayers.Dense(output_feature_count, name="mu_head")(x)
        log_var = TfLayers.Dense(output_feature_count, name="log_var_head")(x)

        gaussian_params = TfLayers.Concatenate(axis=-1, name="gaussian_params")([mu, log_var])

        return TfModel(inputs=x_in, outputs=gaussian_params, name="vanilla_gaussian_transformer")
