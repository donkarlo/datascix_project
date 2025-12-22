from typing import Protocol, List

from mathx.linalg.tensor.vector.vector import Vector


class Interface(Protocol):
    _high_dimension_vectors: List[Vector]
    _reduced_dimension_vectors: List[Vector]
    def get_reduced_dimension_vectors(self) -> List[Vector]:
        ...
    def get_high_dimension_vectors(self) -> List[Vector]:
        ...


