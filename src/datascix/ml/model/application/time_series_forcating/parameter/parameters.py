from datascix.ml.model.parameter.parameters import Parameters as BaseModelParameters
from utilix.data.kind.dic.dic import Dic


class Parameters(BaseModelParameters):
    def __init__(self, parameter_values:Dic):
        BaseModelParameters.__init__(self, parameter_values)