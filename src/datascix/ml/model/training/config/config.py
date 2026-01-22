from utilix.data.kind.dic.dic import Dic


class Config:
    def __init__(self, config_values:Dic):
        self._config_values = config_values

    def get_config_values(self):
        return self._config_values