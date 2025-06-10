from abc import ABC, abstractmethod
import numpy as np
from utilityx.conf.conf import Conf
from utilityx.pythonx.data_type.float_seqx import Seqx

from datascix.ml.seq2seq.trainer import Trainer


class Predictor(ABC):
    def __init__(self, trainer:Trainer, inp_seq:np.ndarray, prd_seq_len:int, confs:Conf):
        self._inp_seq = inp_seq
        self._prd_seq_len = prd_seq_len
        self._confs = confs
        self._trainer = trainer

        # lazy loading
        self._prd_seq = None

    @abstractmethod
    def get_prd_seq(self)->Seqx:
        pass
