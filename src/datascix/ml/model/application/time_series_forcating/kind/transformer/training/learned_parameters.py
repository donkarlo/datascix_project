import numpy as np


class LearnedParameters:
    """
    Things that exist only AFTER training (fit).
    For now: model weights + input feature count (to build the model before set_weights).

    Note:
        - weights are stored as a list of np.ndarray, exactly as Keras get_weights() returns.
        - no normalization stats here.
    """

    def __init__(
            self,
            model_weights: list[np.ndarray] | None = None,
            input_feature_count: int | None = None,
    ):
        self._model_weights = None if model_weights is None else [np.asarray(w) for w in model_weights]
        self._input_feature_count = None if input_feature_count is None else int(input_feature_count)

        if self._input_feature_count is not None and self._input_feature_count <= 0:
            raise ValueError("input_feature_count must be > 0.")

    def has_model_weights(self) -> bool:
        return self._model_weights is not None

    def get_model_weights(self) -> list[np.ndarray]:
        if self._model_weights is None:
            raise ValueError("model_weights is not set.")
        return self._model_weights

    def has_input_feature_count(self) -> bool:
        return self._input_feature_count is not None

    def get_input_feature_count(self) -> int:
        if self._input_feature_count is None:
            raise ValueError("input_feature_count is not set.")
        return self._input_feature_count
