from abc import abstractmethod,ABC
from utilityx.pythonx.data_type.seqx import Seqx
from datascix.ml.supervised.seq2seq.training_conf import TrainingConf


class Model(ABC):
    def __init__(self, sample_seq:Seqx):
        '''
        Just to abstract training
        Args:
            sample_seq
        '''
        self._sample_seq = sample_seq

    @abstractmethod
    def train(self, conf:TrainingConf):
        pass

    @abstractmethod
    def eval_conf(self):
        pass

    @abstractmethod
    def get_prd_seq(self, input_seq:Seqx):
        pass