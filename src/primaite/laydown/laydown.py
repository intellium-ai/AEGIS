from dataclasses import dataclass

from primaite.common.custom_typing import Serializable
from primaite.network import Network
from primaite.nodes import NodeStateInstructionGreen, NodeStateInstructionRed
from primaite.pol.ier import IER


@dataclass
class Laydown(Serializable):
    initial_network: Network
    iers: list[IER]
    red_pols: list[NodeStateInstructionRed]
    green_pols: list[NodeStateInstructionGreen]

    def serialize(self):
        res = [
            *self.initial_network.serialize(),
            *[ier.serialize() for ier in self.iers],
            *[pol.serialize() for pol in self.red_pols + self.green_pols],
        ]

        return res
