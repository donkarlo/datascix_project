from typing import List

from datascix.data.pipe.node.group import Group as NodeGroup
from datascix.data.pipe.node.node import Node


class Pipe:
    def __init__(self, node_group: NodeGroup):
        self._node_group = node_group

    @classmethod
    def init_by_nodes_list(cls, nodes_lits: List[Node]) -> None:
        pass
