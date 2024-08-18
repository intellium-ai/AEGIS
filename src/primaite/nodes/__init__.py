# © Crown-owned copyright 2023, Defence Science and Technology Laboratory UK
"""Nodes represent network hosts in the simulation."""

from typing import Union

from primaite.nodes.active_node import ActiveNode
from primaite.nodes.node import Node
from primaite.nodes.node_state_instruction_green import NodeStateInstructionGreen
from primaite.nodes.node_state_instruction_red import NodeStateInstructionRed
from primaite.nodes.passive_node import PassiveNode
from primaite.nodes.service_node import ServiceNode

NodeUnion = Union[ActiveNode, PassiveNode, ServiceNode]
"""A Union of ActiveNode, PassiveNode, and ServiceNode."""
