from utilityx.pythonx.data_type.number_seqx import NumberSeqx

class Trainer:
    def __init__(self, sample_seq:NumberSeqx, conf:TrainerConf):
        self._confs = conf
        self._sample_seq = sample_seq

    def get_trained_seq2seq(self):
        return self._sample_seq