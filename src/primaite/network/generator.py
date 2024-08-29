from abc import ABC, abstractmethod

# external imports
import networkx as nx
import random
from typing import Dict, List, Optional
from pathlib import Path
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt

# internal imports
from primaite.common.enums import HardwareState, NodeType, Priority, SoftwareState, FileSystemState, IERType, NodePOLType, NodePOLInitiator
from primaite.common.service import Service
from primaite.nodes import NodeUnion, NodeStateInstructionGreen, NodeStateInstructionRed, ActiveNode, PassiveNode, ServiceNode
from primaite.network.network import Network
from primaite.links import Link
from primaite.config.training_config import load
from primaite.laydown.laydown import Laydown
from primaite.pol.ier import IER

class BaseNetworkGenerator(ABC):

    @abstractmethod
    def generate(self, **kwargs) -> Network: ...


class GENINDNetworkGenerator(BaseNetworkGenerator): ...

class NetworkGenerator:

    def __init__ (
            self, 
            training_config_path: str | Path,
            graph_size: int = 30, 
            random_seed: Optional[int] = None, 
            services: List[str] = ["HTTP", "SSH"],
            ports:  List[str] = ["80", "22"] 
        ) -> None:
        """
        Initialise the Node State Instruction for the red agent.

        :param training_config: Path to the training config.
        :param graph_size: Number of nodes in the network
        :param random_seed: The random seed. Defaults to None.
        :param services: List of services (e.g. ["TCP"]).
        :param ports: List of ports (e.g. ["80"]). Should correspond to `services`.
        """
        self.graph_size = graph_size
        self.random_seed = random_seed
        self.services = services
        self.ports = ports
        self.training_config = load(training_config_path)

        self.graph = self.__generate_graph()
        self.links_dict = self.__create_link_dict()
        self.nodes_dict = self.__create_nodes_dict()
        self.network = self.__create_network()

    def __generate_graph(self):

        # generate graph from networkx
        G = nx.random_internet_as_graph(n=self.graph_size, seed=self.random_seed)

        # assigning node types
        for node in G.nodes():
            if G.degree[node] > 5:
                G.nodes[node]['type'] = NodeType.SERVER
            else:
                G.nodes[node]['type'] = random.choice([
                    NodeType.CCTV,
                    NodeType.SWITCH,
                    NodeType.COMPUTER,
                    NodeType.LINK
                ])

        return G
    
    def __create_link_dict(self):
        # init empty link dict
        links_dict: Dict[str, Link] = {}

        # iterate over each edge
        for index, (src_node, dest_node, _) in enumerate(self.graph.edges(data=True)):
            # generate a unique ID for each link
            link_id = f"{index}"
            
            # get the names and IDs of the source and destination nodes
            src_node_id = str(src_node)
            src_node_name = f"Node_{src_node_id}"
            dest_node_id = str(dest_node)
            dest_node_name = f"Node_{dest_node_id}"
            
            # create the Link object
            link = Link(
                _name=link_id,
                _id=link_id,
                _bandwidth=random.randint(100, 1000),  # TODO:(make variable) random bandwidth between 100 and 1000 bps
                _source_node_id=src_node_id,
                _source_node_name=src_node_name,
                _dest_node_id=dest_node_id,
                _dest_node_name=dest_node_name,
                _services=self.services
            )
            
            # add the link to the `links_dict`
            links_dict[link_id] = link

        return links_dict

    def __create_nodes_dict(self):

        # init empty nodes dict
        nodes_dict: Dict[str, NodeUnion] = {}

        # TODO: make variable
        # additional default values for IP address, software state, and file system state 
        default_ip_address = "192.168.1.1"
        default_software_state = SoftwareState.GOOD
        default_file_system_state = FileSystemState.GOOD

        # modify the node creation logic to include the required arguments
        for node_id in self.graph.nodes():
            # extract node attributes
            node_type = self.graph.nodes[node_id]['type']
            node_name = f"Node_{node_id}"
            
            # determine priority and hardware state (for demonstration purposes, these are random)
            priority = random.choice(list(Priority))
            hardware_state = random.choice(list(HardwareState))
            
            # based on the node_type, create an appropriate NodeUnion instance
            if node_type == NodeType.SERVER or node_type == NodeType.COMPUTER:
                node = ServiceNode(
                    node_id=str(node_id),
                    name=node_name,
                    node_type=node_type,
                    priority=priority,
                    hardware_state=hardware_state,
                    ip_address=default_ip_address,
                    software_state=default_software_state,
                    file_system_state=default_file_system_state,
                    config_values=self.training_config
                )

                for i, service in enumerate(self.services):
                    node.add_service(Service(name=service, port=self.ports[i], software_state=default_software_state))

            else:
                node = ActiveNode(
                    node_id=str(node_id),
                    name=node_name,
                    node_type=node_type,
                    priority=priority,
                    hardware_state=hardware_state,  # TODO: check this
                    ip_address=default_ip_address,
                    software_state=default_software_state,
                    file_system_state=default_file_system_state,
                    config_values=self.training_config
                )
            
            # add the node to the `nodes_dict`
            nodes_dict[str(node_id)] = node
        
        return nodes_dict

    def __create_network(self):
        return Network(
            nodes_dict=self.nodes_dict, 
            links_dict=self.links_dict, 
            service_names=self.services, 
            ports_list=self.ports, 
            graph=self.graph
        )

    def show_network(self) -> None:

        # NOTE: not sure the color of links changes with traffic

        pos = nx.spring_layout(self.graph, seed=self.random_seed)

        # assigning colours to nodes
        NODE_TYPE_MAP = {
            NodeType.CCTV: {"colour": "#FFB3FF"},     # Pastel Magenta
            NodeType.SWITCH: {"colour": "#A3C1E0"},   # Pastel Blue
            NodeType.COMPUTER: {"colour": "#CFCFCF"}, # Pastel Gray
            NodeType.LINK: {"colour": "#77DD77"},     # Pastel Green
            NodeType.MONITOR: {"colour": "#FFFFB3"},  # Pastel Yellow
            NodeType.PRINTER: {"colour": "#FFD1A3"},  # Pastel Orange
            NodeType.LOP: {"colour": "#D1B3FF"},      # Pastel Purple
            NodeType.RTU: {"colour": "#D2B48C"},      # Pastel Brown (Tan)
            NodeType.ACTUATOR: {"colour": "#FFB6C1"}, # Pastel Pink
            NodeType.SERVER: {"colour": "#B3FFFF"},   # Pastel Cyan
        }

        node_colors = [NODE_TYPE_MAP[v['type']]["colour"] for _, v in self.graph.nodes(data=True)]

        # get traffic on each edge
        edge_traffic_levels = []
        for index, _ in enumerate(self.graph.edges(data=True)):
            link_id = f"{index}"
            link = self.links_dict[link_id]
            traffic_level = link.get_traffic_level()
            edge_traffic_levels.append(traffic_level)

        # normalize edge traffic levels for better visualization
        norm = Normalize(vmin=min(edge_traffic_levels), vmax=max(edge_traffic_levels))
        edge_colors_normalized = [norm(level) for level in edge_traffic_levels]

        # draw the network with edges colored by traffic level
        nx.draw_networkx(
            self.graph,
            pos=pos,
            node_color=node_colors,
            with_labels=False,
            edge_color=edge_colors_normalized,
            edge_cmap=plt.cm.hot, 
            edge_vmax=1.0, 
        )

        pos_higher = {k: (v[0], v[1]) for k, v in pos.items()}
        nx.draw_networkx_labels(self.graph, pos=pos_higher, font_size=5, font_color="black")

        edge_labels = {(u, v): index for index, (u, v, _) in enumerate(self.graph.edges(data=True))}
        nx.draw_networkx_edge_labels(self.graph, pos=pos, font_size=4, edge_labels=edge_labels, font_color="black")



# # JACK's CODE (RIP SMIT)
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
