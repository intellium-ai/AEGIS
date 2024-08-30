from abc import ABC, abstractmethod

# external imports
import networkx as nx
import random
from typing import Dict, List, Optional, Union
from pathlib import Path
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env file


# internal imports
from primaite.common.enums import HardwareState, NodeType, Priority, SoftwareState, FileSystemState, IERType, NodePOLType, NodePOLInitiator
from primaite.common.service import Service
from primaite.nodes import NodeUnion, NodeStateInstructionGreen, NodeStateInstructionRed, ActiveNode, PassiveNode, ServiceNode
from primaite.agents.llm.observation import network_connectivity_desc
from primaite.network.network import Network
from primaite.links import Link
from primaite.config.training_config import load
from primaite.laydown.laydown import Laydown
from primaite.pol.ier import IER
from primaite.network import QUESTION_VARIATIONS

# TODO: move this somewhere
from openai import OpenAI

class OpenAIClient:
    def __init__(self, openai_api_key: str):
        self.client = OpenAI(api_key=openai_api_key)

    def generate(self, prompt: str, max_new_tokens: int = 1000) -> str:
        messages = [{"role": "user", "content": prompt}]

        response = self.client.chat.completions.create(
            model="gpt-4o", messages=messages, max_tokens=max_new_tokens
        )

        return response.choices[0].message.content


class BaseNetworkGenerator(ABC):

    @abstractmethod
    def generate(self, **kwargs) -> Network: ...


class GENINDNetworkGenerator(BaseNetworkGenerator): ...

class NetworkGenerator:

    def __init__ (
            self,
            training_config_path: Union[str, Path],
            graph_size: int = 30,
            random_seed: Optional[int] = None,
            services: List[str] = ["HTTP", "SSH"],
            ports: List[str] = ["80", "22"],
            question: Optional[str] = None
        ) -> None:
        """
        Initialize the NetworkGenerator.

        Args:
            training_config_path (Union[str, Path]): Path to the training config file.
            graph_size (int, optional): Number of nodes in the network. Defaults to 30.
            random_seed (Optional[int], optional): The random seed. Defaults to None.
            services (List[str], optional): List of services. Defaults to ["HTTP", "SSH"].
            ports (List[str], optional): List of ports corresponding to services. Defaults to ["80", "22"].
            

        Returns:
            None
        """
        self.graph_size: int = graph_size
        self.random_seed: Optional[int] = random_seed
        self.services: List[str] = services
        self.ports: List[str] = ports
        self.training_config = load(training_config_path)
        self.iers: List[IER] = []
        self.green_pols: List[NodeStateInstructionGreen] = []
        self.red_pols: List[NodeStateInstructionRed] = []

        self.graph: nx.Graph = self.__generate_graph()
        self.links_dict: Dict[str, Link] = self.__create_link_dict()
        self.nodes_dict: Dict[str, NodeUnion] = self.__create_nodes_dict()
        self.network: Network = self.__create_network()
        self.network_description = network_connectivity_desc(self.network)
        self.laydown: Optional[Laydown] = None
        self.question: str = question if question else self.__choose_question()
        self.target_answer: str = self.__ask_llm()

    # --------------------- IERS
    def generate_iers(self, count: int = 5) -> None:
        for i in range(count):
            edge = random.choice(list(self.links_dict.values()))
            ier_type = random.choice(list(IERType))
            start_step = random.randint(1, 127)
            end_step = random.randint(start_step, 128)

            self.add_ier(
                _type=ier_type,
                _id=str(i),
                _start_step=start_step,
                _end_step=end_step,
                _load=10000,
                _protocol="HTTP",
                _port=["80"],
                _source_node_id=edge.source_node_id,
                _dest_node_id=edge.dest_node_id,
                _mission_criticality=random.randint(0, 5)
            )

    def add_ier(self, **kwargs) -> None:
        ier = IER(**kwargs)
        self.iers.append(ier)

    def remove_ier(self, index: int) -> None:
        if 0 <= index < len(self.iers):
            del self.iers[index]
        else:
            raise IndexError("IER index out of range")

    def modify_ier(self, index: int, **kwargs) -> None:
        if 0 <= index < len(self.iers):
            ier = self.iers[index]
            for key, value in kwargs.items():
                if hasattr(ier, key):
                    setattr(ier, key, value)
        else:
            raise IndexError("IER index out of range")
        
    def get_iers(self) -> List[IER]:
        return self.iers
    
    def generate_laydown(self, laydown_save_path: Union[str, Path]) -> None:
        self.laydown = Laydown(
            initial_network=self.network,
            iers=self.iers,
            red_pols=self.red_pols,
            green_pols=self.green_pols
        )
        self.laydown.save(laydown_save_path)
    
    # --------------------- POLS
    def generate_green_pols(self, count: int = 5):
        for i in range(count):
            # choose random node
            node = random.choice(list(self.nodes_dict.values()))

            start_step = random.randint(1, 127)
            node_pol_type = random.choice([i for i in list(NodePOLType) if i != NodePOLType.NONE])

            match node_pol_type:
                case NodePOLType.OPERATING:
                    state = random.choice(list(HardwareState))
                case NodePOLType.OS:
                    state = random.choice(list(SoftwareState))
                case NodePOLType.SERVICE:
                    state = random.choice(list(SoftwareState))
                case NodePOLType.FILE:
                    state = random.choice(list(FileSystemState))
                case _:
                    state = None

            self.add_green_pol(
                _id=str(i),
                _start_step=start_step,
                _end_step=random.randint(start_step, 128),
                _node_id=node.node_id,
                _node_pol_type=node_pol_type,
                _service_name="HTTP",
                _state=state
            )

    def add_green_pol(self, **kwargs) -> None:
        pol = NodeStateInstructionGreen(**kwargs)
        self.green_pols.append(pol)

    def get_green_pols(self) -> List[NodeStateInstructionGreen]:
        return self.green_pols
    
    def generate_red_pols(self, count=5):
        for i in range(count):
            # get random edge
            edge = random.choice(list(self.links_dict.values()))

            start_step = random.randint(1, 127)
            pol_type = random.choice([i for i in NodePOLType if i not in [NodePOLType.NONE, NodePOLType.FILE]])

            match pol_type:
                case NodePOLType.OPERATING:
                    pol_state = random.choice(list(HardwareState))
                case NodePOLType.OS:
                    pol_state = random.choice(list(SoftwareState))
                case NodePOLType.SERVICE:
                    pol_state = random.choice(list(SoftwareState))
                case NodePOLType.FILE:
                    pol_state = random.choice(list(FileSystemState))
                case _:
                    pol_state = None

            print("target nide id, ", edge.dest_node_id)

            self.add_red_pol(
                _id=str(i),
                _start_step=start_step,
                _end_step=random.randint(start_step, 128),
                _target_node_id=edge.dest_node_id,
                _pol_initiator=NodePOLInitiator.IER,
                _pol_type=pol_type,
                pol_protocol="HTTP",
                _pol_state=pol_state,
                _pol_source_node_id=edge.source_node_id,
                _pol_source_node_service="HTTP",
                _pol_source_node_service_state=pol_state
            )

    def add_red_pol(self, **kwargs) -> None:
        pol = NodeStateInstructionRed(**kwargs)
        self.red_pols.append(pol)

    def get_red_pols(self) -> List[NodeStateInstructionRed]:
        return self.red_pols

    # --------------------- NETWORK CONFIG

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

    # --------------------- LLM ANSWER
    @staticmethod
    def __choose_question() -> str:
        base_question = random.choice(list(QUESTION_VARIATIONS.keys()))
        question = random.choice(QUESTION_VARIATIONS[base_question])

        return question

    def __ask_llm(self):

        load_dotenv() 
        open_ai_key = os.getenv('OPEN_AI_KEY')

        prompt = f"""
            Your task is to answer the given question about a network using the provided context. Your answer should be very brief, it may even be as simple as a single word or number.
            Question: {self.question}.
            Context:
            {self.network_description}
        """

        openai = OpenAIClient(openai_api_key=open_ai_key)

        answer = openai.generate(prompt=prompt)

        return answer
