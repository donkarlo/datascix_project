from datascix.ml.kind.classification.label import Label


class LabelSet:
    def __init__(self, labels:tuple[Label,...]):
        self._labels = labels
        self.__has_unique_labels()

    def __has_unique_labels(self):
        return False