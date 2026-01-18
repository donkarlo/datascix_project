import numpy as np
import tensorflow as tf

from datascix.ml.kind.sequence_to_sequence.kind.transformer.sliding_window.generator import Generator
from datascix.ml.kind.sequence_to_sequence.kind.transformer.sliding_window.sliding_window import SlidingWindow
from tensorflow.keras import layers as TfLayers, Model as TfModel
from utilix.data.storage.kind.file.numpi.multi_valued import MultiValued as NpMultiValued
from utilix.os.file_system.file.file import File as OsFile
from utilix.os.file_system.path.file import File as FilePath


@tf.keras.utils.register_keras_serializable(package="datascix")
class TransformerWithUncertainty(TfModel):
    """
    Minimal seq2seq Transformer with heteroscedastic Gaussian output.

    Notes:
    - Model output is a single tensor: concat([mean, logvar], axis=-1).
      This avoids Keras multi-output loss mapping issues.
    - Training uses teacher forcing.
    - Inference uses autoregressive decoding and feeds predicted mean.
    """

    def __init__(
            self,
            model_dimension: int,
            number_of_attention_heads: int,
            feed_forward_dimension: int,
            output_time_steps: int,
            output_feature_count: int,
            dropout_rate: float = 0.1,
            maximum_time_steps: int = 2048,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.model_dimension = int(model_dimension)
        self.number_of_attention_heads = int(number_of_attention_heads)
        self.feed_forward_dimension = int(feed_forward_dimension)
        self.output_time_steps = int(output_time_steps)
        self.output_feature_count = int(output_feature_count)
        self.dropout_rate = float(dropout_rate)
        self.maximum_time_steps = int(maximum_time_steps)

        if self.model_dimension % self.number_of_attention_heads != 0:
            raise ValueError("model_dimension must be divisible by number_of_attention_heads.")

        self._per_head_dimension = self.model_dimension // self.number_of_attention_heads

        # Input projections
        self.encoder_input_projection = TfLayers.Dense(self.model_dimension)
        self.decoder_input_projection = TfLayers.Dense(self.model_dimension)

        # Learned positional embeddings
        self.encoder_position_embedding = TfLayers.Embedding(self.maximum_time_steps, self.model_dimension)
        self.decoder_position_embedding = TfLayers.Embedding(self.maximum_time_steps, self.model_dimension)

        # Encoder block (1 layer)
        self.encoder_self_attention = TfLayers.MultiHeadAttention(
            num_heads=self.number_of_attention_heads,
            key_dim=self._per_head_dimension,
            dropout=self.dropout_rate,
        )
        self.encoder_feed_forward = tf.keras.Sequential(
            [
                TfLayers.Dense(self.feed_forward_dimension, activation="relu"),
                TfLayers.Dense(self.model_dimension),
            ]
        )
        self.encoder_dropout_after_attention = TfLayers.Dropout(self.dropout_rate)
        self.encoder_dropout_after_feed_forward = TfLayers.Dropout(self.dropout_rate)
        self.encoder_layer_norm_after_attention = TfLayers.LayerNormalization(epsilon=1e-6)
        self.encoder_layer_norm_after_feed_forward = TfLayers.LayerNormalization(epsilon=1e-6)

        # Decoder block (1 layer)
        self.decoder_self_attention = TfLayers.MultiHeadAttention(
            num_heads=self.number_of_attention_heads,
            key_dim=self._per_head_dimension,
            dropout=self.dropout_rate,
        )
        self.decoder_cross_attention = TfLayers.MultiHeadAttention(
            num_heads=self.number_of_attention_heads,
            key_dim=self._per_head_dimension,
            dropout=self.dropout_rate,
        )
        self.decoder_feed_forward = tf.keras.Sequential(
            [
                TfLayers.Dense(self.feed_forward_dimension, activation="relu"),
                TfLayers.Dense(self.model_dimension),
            ]
        )
        self.decoder_dropout_after_self_attention = TfLayers.Dropout(self.dropout_rate)
        self.decoder_dropout_after_cross_attention = TfLayers.Dropout(self.dropout_rate)
        self.decoder_dropout_after_feed_forward = TfLayers.Dropout(self.dropout_rate)
        self.decoder_layer_norm_after_self_attention = TfLayers.LayerNormalization(epsilon=1e-6)
        self.decoder_layer_norm_after_cross_attention = TfLayers.LayerNormalization(epsilon=1e-6)
        self.decoder_layer_norm_after_feed_forward = TfLayers.LayerNormalization(epsilon=1e-6)

        # Output heads (mean + logvar)
        self.output_projection_mean = TfLayers.Dense(self.output_feature_count)
        self.output_projection_logvar = TfLayers.Dense(self.output_feature_count)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "model_dimension": self.model_dimension,
            "number_of_attention_heads": self.number_of_attention_heads,
            "feed_forward_dimension": self.feed_forward_dimension,
            "output_time_steps": self.output_time_steps,
            "output_feature_count": self.output_feature_count,
            "dropout_rate": self.dropout_rate,
            "maximum_time_steps": self.maximum_time_steps,
        })
        return config

    @classmethod
    def from_config(cls, config: dict) -> "TransformerWithUncertainty":
        return cls(**config)

    def _add_positional_embedding(self, projected: tf.Tensor, position_embedding: TfLayers.Embedding) -> tf.Tensor:
        # projected: (B, T, D)
        time_steps = tf.shape(projected)[1]
        positions = tf.range(start=0, limit=time_steps, delta=1)
        positions = position_embedding(positions)  # (T, D)
        positions = tf.expand_dims(positions, axis=0)  # (1, T, D)
        return projected + positions

    def _encode(self, encoder_input: tf.Tensor, training: bool) -> tf.Tensor:
        # encoder_input: (B, T_in, F_in)
        encoder_projected = self.encoder_input_projection(encoder_input)  # (B, T_in, D)
        encoder_projected = self._add_positional_embedding(encoder_projected, self.encoder_position_embedding)

        attention_output = self.encoder_self_attention(
            query=encoder_projected,
            value=encoder_projected,
            key=encoder_projected,
            training=training,
        )
        attention_output = self.encoder_dropout_after_attention(attention_output, training=training)

        attention_residual = encoder_projected + attention_output
        attention_normalized = self.encoder_layer_norm_after_attention(attention_residual)

        feed_forward_output = self.encoder_feed_forward(attention_normalized, training=training)
        feed_forward_output = self.encoder_dropout_after_feed_forward(feed_forward_output, training=training)

        feed_forward_residual = attention_normalized + feed_forward_output
        encoder_memory = self.encoder_layer_norm_after_feed_forward(feed_forward_residual)
        return encoder_memory  # (B, T_in, D)

    def _decode(self, encoder_memory: tf.Tensor, decoder_input: tf.Tensor, training: bool) -> tf.Tensor:
        # decoder_input: (B, T_out, F_out)
        decoder_projected = self.decoder_input_projection(decoder_input)  # (B, T_out, D)
        decoder_projected = self._add_positional_embedding(decoder_projected, self.decoder_position_embedding)

        # Causal self-attention
        self_attention_output = self.decoder_self_attention(
            query=decoder_projected,
            value=decoder_projected,
            key=decoder_projected,
            use_causal_mask=True,
            training=training,
        )
        self_attention_output = self.decoder_dropout_after_self_attention(self_attention_output, training=training)

        self_attention_residual = decoder_projected + self_attention_output
        self_attention_normalized = self.decoder_layer_norm_after_self_attention(self_attention_residual)

        # Cross-attention
        cross_attention_output = self.decoder_cross_attention(
            query=self_attention_normalized,
            value=encoder_memory,
            key=encoder_memory,
            training=training,
        )
        cross_attention_output = self.decoder_dropout_after_cross_attention(cross_attention_output, training=training)

        cross_attention_residual = self_attention_normalized + cross_attention_output
        cross_attention_normalized = self.decoder_layer_norm_after_cross_attention(cross_attention_residual)

        # Feed-forward
        feed_forward_output = self.decoder_feed_forward(cross_attention_normalized, training=training)
        feed_forward_output = self.decoder_dropout_after_feed_forward(feed_forward_output, training=training)

        feed_forward_residual = cross_attention_normalized + feed_forward_output
        decoder_hidden = self.decoder_layer_norm_after_feed_forward(feed_forward_residual)

        mean = self.output_projection_mean(decoder_hidden)  # (B, T_out, F)
        logvar = self.output_projection_logvar(decoder_hidden)  # (B, T_out, F)

        return tf.concat([mean, logvar], axis=-1)  # (B, T_out, 2F)

    def call(self, inputs, training: bool = False) -> tf.Tensor:
        encoder_input, decoder_input = inputs
        encoder_memory = self._encode(encoder_input, training=training)
        mean_logvar = self._decode(encoder_memory, decoder_input, training=training)
        return mean_logvar

    def _build_teacher_forcing_decoder_input(self, target_tensor: tf.Tensor) -> tf.Tensor:
        batch_size = tf.shape(target_tensor)[0]
        zeros_first = tf.zeros((batch_size, 1, self.output_feature_count), dtype=target_tensor.dtype)
        shifted = target_tensor[:, :-1, :]
        decoder_input = tf.concat([zeros_first, shifted], axis=1)
        return decoder_input

    @tf.keras.utils.register_keras_serializable(package="datascix")
    def gaussian_nll(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        mean, logvar = tf.split(y_pred, num_or_size_splits=2, axis=-1)

        logvar = tf.clip_by_value(logvar, -20.0, 5.0)
        var = tf.exp(logvar)

        nll = 0.5 * (logvar + tf.square(y_true - mean) / var)
        return tf.reduce_mean(nll)

    def train(
            self,
            input_array: np.ndarray,
            target_array: np.ndarray,
            epochs: int = 5,
            batch_size: int = 16,
            learning_rate: float = 1e-3,
            shuffle: bool = True,
    ) -> None:

        if not isinstance(input_array, np.ndarray) or not isinstance(target_array, np.ndarray):
            raise TypeError("input_array and target_array must be np.ndarray.")

        if input_array.ndim != 3 or target_array.ndim != 3:
            raise ValueError("input_array and target_array must have shape (B, T, F).")

        if input_array.shape[0] != target_array.shape[0]:
            raise ValueError("Batch size mismatch between input_array and target_array.")

        if target_array.shape[1] != self.output_time_steps:
            raise ValueError(f"target_array time steps must be {self.output_time_steps}, got {target_array.shape[1]}.")

        if target_array.shape[2] != self.output_feature_count:
            raise ValueError(
                f"target_array feature count must be {self.output_feature_count}, got {target_array.shape[2]}."
            )

        encoder_input = tf.convert_to_tensor(input_array.astype(np.float32))
        target_tensor = tf.convert_to_tensor(target_array.astype(np.float32))
        decoder_input = self._build_teacher_forcing_decoder_input(target_tensor)

        _ = self((encoder_input[:1], decoder_input[:1]), training=False)

        self.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=self.gaussian_nll,
        )

        self.fit(
            (encoder_input, decoder_input),
            target_tensor,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=shuffle,
        )

    def predict_autoregressive(self, input_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(input_array, np.ndarray):
            raise TypeError("input_array must be np.ndarray.")
        if input_array.ndim != 3:
            raise ValueError("input_array must have shape (B, T_in, F_in).")

        encoder_input = tf.convert_to_tensor(input_array.astype(np.float32))
        encoder_memory = self._encode(encoder_input, training=False)

        batch_size = tf.shape(encoder_input)[0]
        mean_sequence = tf.zeros((batch_size, 0, self.output_feature_count), dtype=tf.float32)
        var_sequence = tf.zeros((batch_size, 0, self.output_feature_count), dtype=tf.float32)

        step = 0
        while step < self.output_time_steps:
            if step == 0:
                current_decoder_input = tf.zeros((batch_size, 1, self.output_feature_count), dtype=tf.float32)
            else:
                zeros_first = tf.zeros((batch_size, 1, self.output_feature_count), dtype=tf.float32)
                current_decoder_input = tf.concat([zeros_first, mean_sequence], axis=1)
                current_decoder_input = current_decoder_input[:, :step + 1, :]

            mean_logvar = self._decode(encoder_memory, current_decoder_input, training=False)
            mean, logvar = tf.split(mean_logvar, num_or_size_splits=2, axis=-1)

            next_mean = mean[:, -1:, :]
            next_logvar = tf.clip_by_value(logvar[:, -1:, :], -20.0, 5.0)
            next_var = tf.exp(next_logvar)

            mean_sequence = tf.concat([mean_sequence, next_mean], axis=1)
            var_sequence = tf.concat([var_sequence, next_var], axis=1)
            step += 1

        return mean_sequence.numpy(), var_sequence.numpy()

    def save_model(self, save_path: str) -> None:
        self.save(save_path)

    @staticmethod
    def load_transformer_model(save_path: str) -> "TransformerWithUncertainty":
        loaded_model = tf.keras.models.load_model(
            save_path,
            custom_objects={
                "TransformerWithUncertainty": TransformerWithUncertainty,
                "gaussian_nll": TransformerWithUncertainty.gaussian_nll,
            },
            compile=False,
        )
        return loaded_model


if __name__ == "__main__":
    file_path = FilePath(
        "/home/donkarlo/Dropbox/phd/data/experiements/oldest/robots/uav1/structure/mind/memory/explicit/long_term/episodic/normal/gaussianed_quaternion_kinematic_sliced_from_1_to_300000/time_position_sequence_sliced_from_1_to_300000.npz"
    )
    os_file = OsFile.init_from_path(file_path)
    storage = NpMultiValued(os_file, False)
    storage.load()

    ram = storage.get_ram()[0:30000].astype(np.float32)

    sliding_window = SlidingWindow(100, 100, 5)
    sliding_windows_generator = Generator(ram, sliding_window)

    input_array = sliding_windows_generator.get_inputs()
    output_array = sliding_windows_generator.get_outputs()

    transformer_model = TransformerWithUncertainty(
        model_dimension=128,
        number_of_attention_heads=8,
        feed_forward_dimension=256,
        output_time_steps=sliding_window.get_output_length(),
        output_feature_count=output_array.shape[2],
        dropout_rate=0.1,
    )



    transformer_model.train(input_array, output_array, epochs=3, batch_size=8)

    transformer_model.save_model("transformer_seq2seq_uncertainty_saved_model.keras")

    reloaded_model = TransformerWithUncertainty.load_transformer_model(
        "transformer_seq2seq_uncertainty_saved_model.keras")

    predicted_mean, predicted_var = reloaded_model.predict_autoregressive(input_array[0:100])
    print("predicted_mean shape:", predicted_mean)
    # print("predicted_var shape:", predicted_var)
