from abc import ABC, abstractmethod
from utilityx.pythonx.data_type.seqx import Seqx


class Seq2Seq(ABC):
    def __init__(self):
        '''
        One object for each predictor
        '''

    @abstractmethod
    def get_prd_seq(self, inp_seq:Seqx)->Seqx:
        pass

    @abstractmethod
    def train(self, sample_seq:Seqx):
        pass
