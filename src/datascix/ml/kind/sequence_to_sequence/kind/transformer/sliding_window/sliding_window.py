class SlidingWindow:
    def __init__(self, input_length: int, output_length: int, step: int) -> None:
        if input_length <= 0:
            raise ValueError("input_length must be a positive integer")
        if output_length <= 0:
            raise ValueError("output_length must be a positive integer")
        if step <= 0:
            raise ValueError("step must be a positive integer")

        self._input_length = input_length
        self._output_length = output_length
        self._step = step

    def get_input_length(self) -> int:
        return self._input_length

    def get_output_length(self) -> int:
        return self._output_length

    def get_step(self) -> int:
        return self._step