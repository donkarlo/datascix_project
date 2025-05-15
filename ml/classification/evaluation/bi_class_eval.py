class BiClassEval:
    def __init__(self, tp:int, tn:int, fp:int, fn:int):
        self._tp = tp
        self._tn = tn
        self._fp = fp
        self._fn = fn

    def get_accuracy(self):
        total = self._tp + self._tn + self._fp + self._fn
        return (self._tp + self._tn) / total if total != 0 else 0

    def get_precision(self):
        denom = self._tp + self._fp
        return self._tp / denom if denom != 0 else 0

    def get_recall(self):
        denom = self._tp + self._fn
        return self._tp / denom if denom != 0 else 0

    def get_f1_score(self):
        precision = self.get_precision()
        recall = self.get_recall()
        denom = precision + recall
        return 2 * precision * recall / denom if denom != 0 else 0

    def get_confusion_matrix(self):
        return [[self._tp, self._fn], [self._fp, self._tn]]

