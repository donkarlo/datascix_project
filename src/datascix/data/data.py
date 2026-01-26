import numpy as np
from typing import Any

class Data:
    """
    Here group is only numerical numpy array so it is different than group in utilix.group
    """
    def __init__(self, data:Any):
        self._np_array = None
        if isinstance(data, np.ndarray):
            self._np_array = data

    def get_np_array(self) -> np.ndarray:
        return self._np_array

