import numpy as np

class LearnedParameters:
    def __init__(
            self,
            input_mean: np.ndarray | None = None,
            input_std: np.ndarray | None = None,
            target_mean: np.ndarray | None = None,
            target_std: np.ndarray | None = None,
    ):
        self._input_mean = None if input_mean is None else np.asarray(input_mean, dtype=np.float32)
        self._input_std = None if input_std is None else np.asarray(input_std, dtype=np.float32)
        self._target_mean = None if target_mean is None else np.asarray(target_mean, dtype=np.float32)
        self._target_std = None if target_std is None else np.asarray(target_std, dtype=np.float32)

        if self._input_mean is not None and self._input_std is not None:
            if self._input_mean.shape != self._input_std.shape:
                raise ValueError("input_mean and input_std must have the same shape.")

        if self._target_mean is not None and self._target_std is not None:
            if self._target_mean.shape != self._target_std.shape:
                raise ValueError("target_mean and target_std must have the same shape.")

    def has_input_normalization(self) -> bool:
        return self._input_mean is not None and self._input_std is not None

    def has_target_normalization(self) -> bool:
        return self._target_mean is not None and self._target_std is not None

    def get_input_mean(self) -> np.ndarray:
        if self._input_mean is None:
            raise ValueError("input_mean is not set.")
        return self._input_mean

    def get_input_std(self) -> np.ndarray:
        if self._input_std is None:
            raise ValueError("input_std is not set.")
        return self._input_std

    def get_target_mean(self) -> np.ndarray:
        if self._target_mean is None:
            raise ValueError("target_mean is not set.")
        return self._target_mean

    def get_target_std(self) -> np.ndarray:
        if self._target_std is None:
            raise ValueError("target_std is not set.")
        return self._target_std