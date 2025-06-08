import numpy as np


class Seq2Seq:
    def __init__(self, inp_seq:np.ndarray):
        self._inp_seq = inp_seq