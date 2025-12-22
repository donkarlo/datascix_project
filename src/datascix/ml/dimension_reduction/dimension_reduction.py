from typing import List

from datascix.ml.dimension_reduction.interface import Interface
from mathx.linalg.tensor.vector.vector import Vector


class DimentionReduction(Interface):
    def get_reduced_dimension_vectors(self) -> List[Vector]:
        return super().get_reduced_dimension_vectors()

    def get_high_dimension_vectors(self) -> List[Vector]:
        return super().get_high_dimension_vectors()