# file: datascix/ml/model/application/sequence_to_sequence/kind/time_series/kind/transformer/architecture/gaussian_architecture.py
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers as TfLayers, Model as TfModel

from datascix.ml.model.application.sequence_to_sequence.kind.time_series.kind.transformer.architecture.architecture import \
    Architecture as BaseTransformerArchitecture


class Architecture(BaseTransformerArchitecture):
    """Build a vanilla direct seq2seq Transformer that outputs Gaussian parameters.

    Output heads:
        - mu:      (B, T_out, F_out)
        - log_var: (B, T_out, F_out)  (diagonal variance, log-space)
    """

    def build_tf_model(self) -> TfModel:
        d_model = int(self.get_model_dimension())
        heads = int(self.get_number_of_attention_heads())
        ff_dim = int(self.get_feed_forward_dimension())
        input_feature_count = int(self.get_input_feature_count())
        output_time_steps = int(self.get_output_time_steps())
        output_feature_count = int(self.get_output_feature_count())
        maximum_time_steps = int(self.get_maximum_time_steps())
        dropout_rate = float(self.get_dropout_rate())

        if d_model % heads != 0:
            raise ValueError("model_dimension must be divisible by number_of_attention_heads.")
        per_head = d_model // heads

        x_in = TfLayers.Input(shape=(output_time_steps, input_feature_count), dtype=tf.float32, name="x_in")

        x = TfLayers.Dense(d_model, name="in_proj")(x_in)

        pos_emb = TfLayers.Embedding(maximum_time_steps, d_model, name="pos_emb")

        def add_positional(tensor: tf.Tensor) -> tf.Tensor:
            time_steps = tf.shape(tensor)[1]
            positions = pos_emb(tf.range(time_steps))
            positions = tf.expand_dims(positions, axis=0)
            return tensor + positions

        x = TfLayers.Lambda(add_positional, name="add_pos")(x)

        mha = TfLayers.MultiHeadAttention(num_heads=heads, key_dim=per_head, dropout=dropout_rate, name="mha")
        attn = mha(query=x, value=x, key=x)
        attn = TfLayers.Dropout(dropout_rate, name="drop_after_attn")(attn)
        x = TfLayers.LayerNormalization(epsilon=1e-6, name="ln_after_attn")(x + attn)

        ff = tf.keras.Sequential([TfLayers.Dense(ff_dim, activation="relu"), TfLayers.Dense(d_model)], name="ffn")(x)
        ff = TfLayers.Dropout(dropout_rate, name="drop_after_ffn")(ff)
        x = TfLayers.LayerNormalization(epsilon=1e-6, name="ln_after_ffn")(x + ff)

        mu = TfLayers.Dense(output_feature_count, name="mu_head")(x)
        log_var = TfLayers.Dense(output_feature_count, name="log_var_head")(x)

        return TfModel(inputs=x_in, outputs=[mu, log_var], name="vanilla_gaussian_transformer")
