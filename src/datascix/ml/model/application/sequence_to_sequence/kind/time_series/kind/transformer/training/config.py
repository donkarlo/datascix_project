class Config:
    def __init__(
            self,
            epochs: int,
            batch_size: int,
            learning_rate: float = 1e-3,
            shuffle: bool = True,
    ):

        self._epochs = int(epochs)


        self._batch_size = int(batch_size)
        self._learning_rate = float(learning_rate)
        self._shuffle = bool(shuffle)

        if self._epochs <= 0:
            raise ValueError("epochs must be > 0.")
        if self._batch_size <= 0:
            raise ValueError("batch_size must be > 0.")
        if self._learning_rate <= 0.0:
            raise ValueError("learning_rate must be > 0.")


    def get_epochs(self) -> int:
        return self._epochs


    def get_batch_size(self) -> int:
        return self._batch_size


    def get_learning_rate(self) -> float:
        return self._learning_rate


    def get_shuffle(self) -> bool:
        return self._shuffle