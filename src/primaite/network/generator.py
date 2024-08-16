from abc import ABC, abstractmethod

from primaite.network.network import Network


class BaseNetworkGenerator(ABC):

    @abstractmethod
    def generate(self, **kwargs) -> Network: ...


class GENINDNetworkGenerator(BaseNetworkGenerator): ...


# class NetworkGenerator:

#     services: List[Service]
#     ports: List[int]
#     nodes: List[Node]
#     links: List[Link]
#     node_idx: int

#     def __init__(self) -> None:
#         self.node_idx = 1

#     def add_port_and_service(self, low=1, high=99) -> None:

#         self.services.append(list(Service)[np.random.randint(low=0, high=len(list(Service)))])
#         self.ports.append(np.random.randint(low=low, high=high))

#     def add_node(self) -> None:
#         self.nodes = []

#         node_types = list(NodeType)
#         node_type = node_types[np.random.randint(low=0, high=len(node_types))]

#         sampler = random.sample

#         node_service = NodeService(
#             name=sampler(self.services, 1),
#             port=self.ports[0],
#             state=SoftwareState.GOOD,
#         )

#         node = Node(
#             node_id=self.node_idx,
#             name="PC",
#             node_type=node_type,
#             priority=Priority.P1,
#             hardware_state=HardwareState.ON,
#             ip_address=f"192.168.0.{self.node_idx}",
#             software_state=SoftwareState.GOOD,
#             file_system_state=FileSystemState.GOOD,
#             services=[node_service],
#         )

#         self.nodes.append(node)

#         self.node_idx += 1

#     def _ports_to_dict(self) -> Dict[str, Any]:
#         port_config = [{"port": f"{port}"} for port in self.ports]
#         return {"item_type": "PORTS", "ports_list": port_config}

#     def _services_to_dict(self) -> Dict[str, Any]:
#         service_config = [{"name": service.value} for service in self.services]
#         return {"item_type": "SERVICES", "service_list": service_config}

#     def _nodes_to_dict(self) -> List[Dict[str, Any]]:
#         return [node.dict() for node in self.nodes]

#     def _links_to_dict(self) -> List[Dict[str, Any]]:
#         return [link.dict() for link in self.links]

#     def dict(self) -> List[Dict[str, Any]]:
#         config = []
#         config.append(self._ports_to_dict())
#         config.append(self._services_to_dict())
#         config.extend(self._nodes_to_dict())
#         config.extend(self._links_to_dict())

#         return config

#     def save(self, path: str | Path = "test.yaml") -> None:

#         with open(path, "w+") as f:
#             yaml.safe_dump(self.dict(), f)


# def main():

#     network = NetworkGenerator()
#     network.save("src/primaite/config/_package_data/lay_down/test.yaml")


# if __name__ == "__main__":
#     main()
