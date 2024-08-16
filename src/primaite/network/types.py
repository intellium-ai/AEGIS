from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Final, List

from primaite.common.enums import FileSystemState, HardwareState, NodeType, Priority, SoftwareState


class Service(Enum):
    TCP = "TCP"
    UDP = "UDP"
    TCP_SQL = "TCP_SQL"


@dataclass
class NodeService(Serializable):

    def __init__(
        self,
        name: Service,
        port: int,
        state: SoftwareState,
    ) -> None:
        self.name = name.name
        self.port = port
        self.state = state.name

    def dict(self):
        return self.__dict__


@dataclass
class Node(Serializable):
    """
    - item_type: NODE
        node_id: '2'
        name: CLIENT_2
        node_class: SERVICE
        node_type: COMPUTER
        priority: P5
        hardware_state: 'ON'
        ip_address: 192.168.10.12
        software_state: GOOD
        file_system_state: GOOD
        services:
            - name: TCP
            port: '80'
            state: GOOD
    """

    def __init__(
        self,
        node_id: int,
        name: str,
        node_type: NodeType,
        priority: Priority,
        hardware_state: HardwareState,
        ip_address: str,
        software_state: SoftwareState,
        file_system_state: FileSystemState,
        services: List[NodeService],
    ) -> None:
        self.item_type: Final[str] = "NODE"
        self.node_id = str(node_id)
        self.name = name
        self.node_class: Final[str] = "ACTIVE"
        self.priority = priority.name
        self.node_type = node_type.name
        self.hardware_state = hardware_state.name
        self.ip_address = ip_address
        self.software_state = software_state.name
        self.file_system_state = file_system_state.name
        self.services = services

    def serialize(self):
        state_dict = {}
        for k, v in self.__dict__.items():
            if not isinstance(v, list):
                state_dict[k] = v
            else:
                state_dict[k] = [s.dict() for s in v]

        return state_dict


@dataclass
class Link(Serializable):
    """
    - item_type: LINK
        id: '10'
        name: LINK_1
        bandwidth: 1000000000
        source: '1'
        destination: '3'
    """

    def __init__(
        self,
        id: int,
        name: str,
        source: int,
        destination: int,
        bandwidth: int = 100000000,
    ):
        self.item_type: Final[str] = "LINK"
        self.id = str(id)
        self.name = name
        self.source = str(source)
        self.destination = str(destination)
        self.bandwidth = bandwidth

    def dict(self):
        return self.__dict__


@dataclass
class GreenIER(Serializable):
    """
    - item_type: GREEN_IER
        id: '20'
        start_step: 1
        end_step: 256
        load: 10000
        protocol: TCP
        port: '80'
        source: '7'
        destination: '1'
        mission_criticality: 5
    """

    def __init__(
        self,
        id: int,
        start_step: int,
        end_step: int,
        load: int,
        protocol: Service,
        port: int,
        source: int,
        destination: int,
        mission_criticality: int,
    ) -> None:
        self.item_type: Final[str] = "GREEN_IER"
        self.id = str(id)
        self.start_step = start_step
        self.end_step = end_step
        self.load = load
        self.protocol = protocol.name
        self.port = str(port)
        self.source = str(source)
        self.destination = str(destination)
        self.mission_criticality = mission_criticality

    def dict(self):
        return self.__dict__
