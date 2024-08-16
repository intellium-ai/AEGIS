from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Union

import yaml

from primaite.nodes.active_node import ActiveNode
from primaite.nodes.passive_node import PassiveNode
from primaite.nodes.service_node import ServiceNode

NodeUnion = Union[ActiveNode, PassiveNode, ServiceNode]
"""A Union of ActiveNode, PassiveNode, and ServiceNode."""


class Serializable(ABC):

    @abstractmethod
    def serialize(self) -> list[dict] | dict[str, Any]: ...

    def save(self, path: str | Path):
        with open(path, "w+") as f:
            yaml.safe_dump(self.serialize(), f)
