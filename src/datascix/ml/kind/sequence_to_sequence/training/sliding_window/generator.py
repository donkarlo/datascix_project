import numpy as np
from typing import Tuple

from datascix.data.data import Data
from utilix.data.kind.countable.seq.sliding_window.sliding_window import SlidingWindow


class Generator:

    def __init__(self, data: Data, sliding_window: SlidingWindow) -> None:
        self._np_array = data.get_np_array()
        self._sliding_window = sliding_window

    def get_sliding_windows(self) -> Tuple[np.ndarray, np.ndarray]:
        input_length = self._sliding_window.get_input_length()
        output_length = self._sliding_window.get_output_length()
        step = self._sliding_window.get_stride()

        total_length = input_length + output_length
        time_length = self._np_array.shape[0]

        if total_length > time_length:
            raise ValueError("input_length + output_length must be <= data.shape[0]")

        start_indices = self._get_start_indices(time_length, total_length, step)

        inputs = []
        targets = []

        for start_index in start_indices:
            mid_index = start_index + input_length
            end_index = start_index + total_length

            input_window = self._np_array[start_index:mid_index]
            target_window = self._np_array[mid_index:end_index]

            inputs.append(input_window)
            targets.append(target_window)

        input = np.stack(inputs, axis=0)
        output = np.stack(targets, axis=0)

        return (input, output)

    def _get_start_indices(self, time_length: int, total_length: int, stride: int) -> np.ndarray:
        last_start_index = time_length - total_length
        return np.arange(0, last_start_index + 1, stride)