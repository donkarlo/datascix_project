import numpy as np

from datascix.ml.model.supervision.kind.supervion_dependent.kind.self_supervised.sequence_to_sequence.sliding_window.sliding_window import SlidingWindow


class Generator:

    def __init__(self, data: np.ndarray, sliding_window: SlidingWindow) -> None:
        self._np_array = data
        self._sliding_window = sliding_window

        self._inputs = None
        self._outputs = None
        self._input_output_pairs = None

    def get_inputs(self)->np.ndarray:
        if self._inputs is None:
            self._build_input_outputs()
        return self._inputs

    def get_outputs(self)->np.ndarray:
        if self._outputs is None:
            self._build_input_outputs()
        return self._outputs

    def get_input_output_pairs(self)->np.ndarray:
        if self._input_output_pairs is None:
            self._input_output_pairs = np.stack([self.get_inputs(), self.get_outputs()], axis=1)
        return self._input_output_pairs

    def _build_input_outputs(self) -> None:
        input_length = self._sliding_window.get_input_length()
        output_length = self._sliding_window.get_output_length()
        step = self._sliding_window.get_step()

        total_length = input_length + output_length
        time_length = self._np_array.shape[0]

        if total_length > time_length:
            raise ValueError("input_length + output_length must be <= pair_set.shape[0]")

        start_indices = self._get_start_indices(time_length, total_length, step)

        inputs = []
        outputs = []

        for start_index in start_indices:
            mid_index = start_index + input_length
            end_index = start_index + total_length

            current_input_window = self._np_array[start_index:mid_index]
            current_target_window = self._np_array[mid_index:end_index]

            inputs.append(current_input_window)
            outputs.append(current_target_window)

        self._inputs = np.asarray(inputs)
        self._outputs = np.asarray(outputs)

    def _get_start_indices(self, time_length: int, total_length: int, stride: int) -> np.ndarray:
        last_start_index = time_length - total_length
        return np.arange(0, last_start_index + 1, stride)