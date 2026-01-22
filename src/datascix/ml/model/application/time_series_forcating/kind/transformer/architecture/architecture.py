class Architecture:
    def __init__(
            self,
            model_dimension: int,
            number_of_attention_heads: int,
            feed_forward_dimension: int,
            output_time_steps: int,
            output_feature_count: int,
            maximum_time_steps: int = 2048,
            dropout_rate: float = 0.1,
    ):
        self._model_dimension = int(model_dimension)
        self._number_of_attention_heads = int(number_of_attention_heads)
        self._feed_forward_dimension = int(feed_forward_dimension)
        self._output_time_steps = int(output_time_steps)
        self._output_feature_count = int(output_feature_count)
        self._maximum_time_steps = int(maximum_time_steps)
        self._dropout_rate = float(dropout_rate)

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

    def get_output_time_steps(self) -> int:
        return self._output_time_steps

    def get_output_feature_count(self) -> int:
        return self._output_feature_count

    def get_maximum_time_steps(self) -> int:
        return self._maximum_time_steps

    def get_dropout_rate(self) -> float:
        return self._dropout_rate