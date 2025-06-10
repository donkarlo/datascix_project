from utilityx.pythonx.data_type.float_seqx import Seqx


class Eval:
    def __init__(self, prd_seq:Seqx, real_future_seq):
        self._prd_seq = prd_seq
        self._real_future_seq = real_future_seq