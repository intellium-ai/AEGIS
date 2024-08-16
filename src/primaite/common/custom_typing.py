from abc import ABC, abstractmethod
from typing import Any, Union

from primaite.nodes.active_node import ActiveNode
from primaite.nodes.passive_node import PassiveNode
from primaite.nodes.service_node import ServiceNode

NodeUnion = Union[ActiveNode, PassiveNode, ServiceNode]
"""A Union of ActiveNode, PassiveNode, and ServiceNode."""


class Serializable(ABC):

    @abstractmethod
    def dict(self) -> dict[str, Any]: ...
