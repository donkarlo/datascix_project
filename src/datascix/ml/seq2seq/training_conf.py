from utilityx.conf.conf import Conf

class TrainingConf:
    def __init__(self, inp_seq_len:int, out_seq_len:int):
        self._inp_seq_len = inp_seq_len
        self._out_seq_len = out_seq_len
