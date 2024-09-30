from typing import Dict, List, Optional, Union
from pathlib import Path
import random
import os

from matplotlib.colors import Normalize
import networkx as nx
import matplotlib.pyplot as plt
import torch
from dotenv import load_dotenv
from torch_geometric.data import Data

from primaite.primaite_session import PrimaiteSession
from primaite.agents.utils import from_networkx, prepare_graph
from primaite.agents.aegis.modules.openai import OpenAIClient
from primaite.common.simulation import Simulation
from primaite.common.enums import HardwareState, NodeType, Priority, SoftwareState, FileSystemState, IERType, NodePOLType, NodePOLInitiator
from primaite.network.prompting import AgentNodeAction, AgentReasoningNodeSelection, REASON_ACTION_SPACE_NODE_SELECT_TEMPLATE, ACTION_SELECTION_TEMPLATE, ACTION_INFO
from primaite.common.service import Service
from primaite.nodes import NodeUnion, NodeStateInstructionGreen, NodeStateInstructionRed, ActiveNode, ServiceNode
from primaite.agents.vanilla_llm.observation import network_connectivity_desc
from primaite.network.network import Network
from primaite.links import Link
from primaite.config.training_config import TrainingConfig, load
from primaite.laydown.laydown import Laydown
from primaite.pol.ier import IER

load_dotenv()

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

SERVICE_TO_PORT_MAP = {
    "HTTP": "80",
    "SSH": "22",
    "FTP": "21"
}
class NetworkGenerator:

    def __init__ (
            self,
            training_config_path: Union[str, Path],
            graph_size: int = 30,
            random_seed: Optional[int] = None,
            services: List[str] = ["HTTP", "SSH"],
            ports: List[str] = ["80", "22"],
            pretrained_llm: Optional[Laydown] = None
        ) -> None:
        """
        Initialise the NetworkGenerator.

        Args:
            training_config_path (Union[str, Path]): Path to the training config file.
            graph_size (int, optional): Number of nodes in the network. Defaults to 30.
            random_seed (int, optional): The random seed. Defaults to None.
            services (List[str], optional): List of services. Defaults to ["HTTP", "SSH"].
            ports (List[str], optional): List of ports corresponding to services. Defaults to ["80", "22"].
            pretrained_llm (OpenAIClient, optional): An instance of OpenAIClient for using pretrained language models. Defaults to None.
            

        Returns:
            None
        """
        self.training_config_path: str | Path = training_config_path
        self.graph_size: int = graph_size
        self.random_seed: int | None = random_seed
        self.services: list[str] = services
        self.ports: list[str] = ports
        self.training_config: TrainingConfig = load(training_config_path) 
        self.pretrained_llm: None | OpenAIClient = pretrained_llm
        self.iers: list[IER] = []
        self.green_pols: list[NodeStateInstructionGreen] = []
        self.red_pols: list[NodeStateInstructionRed] = []
        self.node_type_counts: dict = {}

        self.graph: nx.Graph = self.__generate_graph()
        self.nodes_dict: dict[str, NodeUnion] = self.__create_nodes_dict()
        self.links_dict: dict[str, Link] = self.__create_link_dict()
        self.network: Network = self.__create_network()
        self.network_description: str = network_connectivity_desc(self.network)
        self.laydown_save_path: str | Path | None = None
        self.laydown: Laydown | None = None
        self.target_reasoning: AgentReasoningNodeSelection | None = None
        self.target_action: AgentNodeAction | None = None
        self.simulation: Simulation | None = None
        self.graph_as_red_and_blue: plt.Figure | None = None

        if self.random_seed is not None:
            random.seed(self.random_seed)


    # --------------------- NETWORK CONFIG

    def __generate_graph(self) -> nx.Graph:
        """
        The `random_internet_as_graph` from `networkx` generates an undirected graph resembling the Internet AS network.

        Node Types:
        * T  -> tier-1
        * M  -> mid-level
        * C  -> customer
        * CP -> content-provider

        Each edge models an ADV communication link with a type and customer attribute.

        [Here are the docs](https://networkx.org/documentation/stable/reference/generated/networkx.generators.internet_as_graphs.random_internet_as_graph.html#random-internet-as-graph)
        """

        # generate graph from networkx
        G = nx.random_internet_as_graph(n=self.graph_size, seed=self.random_seed)

        # find the highest degree node
        highest_degree_node = max(G.nodes(), key=lambda n: G.degree[n])
        # assigning node types
        for node in G.nodes():
            # ensure at least one node is a server
            if node == highest_degree_node:
                G.nodes[node]['type'] = NodeType.SERVER

            elif G.degree[node] > 5:
                G.nodes[node]['type'] = NodeType.SERVER
                
            else:
                # If there are not yet any computer nodes, make this node a computer
                if len([n for n in G.nodes() if G.nodes[n].get('type') == NodeType.COMPUTER]) == 0:
                    G.nodes[node]['type'] = NodeType.COMPUTER
                else:
                    G.nodes[node]['type'] = random.choice([
                        NodeType.SWITCH,
                        NodeType.COMPUTER,
                    ])
                
                    
        return G
    
    def __create_link_dict(self) -> dict[str, Link]:
        # init empty link dict
        links_dict: Dict[str, Link] = {}

        # iterate over each edge
        for index, (src_node, dest_node, _) in enumerate(self.graph.edges(data=True)):
            # generate a unique ID for each link
            link_id = f"{index}"
            

            # get the names and IDs of the source and destination nodes
            src_node_id = str(src_node)
            src_node_name = self.nodes_dict[src_node_id].name
            dest_node_id = str(dest_node)
            dest_node_name = self.nodes_dict[dest_node_id].name
            
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

    def __create_nodes_dict(self) -> dict[str, NodeUnion]:

        # init empty nodes dict
        nodes_dict: Dict[str, NodeUnion] = {}
        services_used = {service: 0 for service in self.services}

        # TODO: make variable
        # additional default values for IP address, software state, and file system state 
        default_ip_address = "192.168.1.1"

        # modify the node creation logic to include the required arguments
        for node_id in self.graph.nodes():
            # extract node attributes
            node_type = self.graph.nodes[node_id]['type']

            # format the nodes name
            if node_type.name not in self.node_type_counts:
                self.node_type_counts[node_type.name] = 1
            else:
                self.node_type_counts[node_type.name] += 1

            node_name = f"{node_type.name}_{self.node_type_counts[node_type.name]}"
            
            # determine priority and hardware state (for demonstration purposes, these are random)
            priority = random.choice(list(Priority))
            file_system_state = random.choice(list(FileSystemState))
            software_state = random.choice(list(SoftwareState))

            # heavily prioritize ON state
            hardware_states = list(HardwareState)
            weights = [0.9 if state == HardwareState.ON else 0.02 for state in hardware_states]
            hardware_state = random.choices(hardware_states, weights=weights)[0]
            
            # based on the node_type, create an appropriate NodeUnion instance
            if node_type == NodeType.SERVER or node_type == NodeType.COMPUTER:
                node = ServiceNode(
                    node_id=str(node_id),
                    name=node_name,
                    node_type=node_type,
                    priority=priority,
                    hardware_state=hardware_state,
                    ip_address=default_ip_address,
                    software_state=software_state,
                    file_system_state=file_system_state,
                    config_values=self.training_config
                )
                
                # ensure each service is used at least once
                unused_services = [s for s, count in services_used.items() if count == 0]
                if unused_services:
                    chosen_services = [random.choice(unused_services)]
                    available_services = set(self.services) - set(chosen_services)
                else:
                    chosen_services = []
                    available_services = set(self.services)
                
                # add more services randomly
                number_of_additional_services = random.randint(0, len(available_services))
                chosen_services.extend(random.sample(available_services, number_of_additional_services))
                
                for service in chosen_services:
                    node.add_service(Service(
                        name=service,
                        port=SERVICE_TO_PORT_MAP[service],
                        software_state=random.choice(list(SoftwareState))
                    ))

            else:
                node = ActiveNode(
                    node_id=str(node_id),
                    name=node_name,
                    node_type=node_type,
                    priority=priority,
                    hardware_state=hardware_state,
                    ip_address=default_ip_address,
                    software_state=software_state,
                    file_system_state=file_system_state,
                    config_values=self.training_config
                )
            
            # add the node to the `nodes_dict`
            nodes_dict[str(node_id)] = node
        
        return nodes_dict

    def __create_network(self) -> Network:

        network = Network(
            nodes_dict=self.nodes_dict, 
            links_dict=self.links_dict, 
            service_names=self.services, 
            ports_list=self.ports, 
            graph=self.graph
        )

        return network
        

    def show_network(self) -> plt.Figure:

        # NOTE: not sure the color of links changes with traffic

        pos = nx.spring_layout(self.graph, seed=self.random_seed)

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

        # Create a new figure
        fig, ax = plt.subplots(figsize=(12, 8))

        # draw the network with edges colored by traffic level
        nx.draw_networkx(
            self.graph,
            pos=pos,
            node_color=node_colors,
            with_labels=False,
            edge_color=edge_colors_normalized,
            edge_cmap=plt.cm.hot, 
            edge_vmax=1.0,
            ax=ax
        )

        pos_higher = {k: (v[0], v[1]) for k, v in pos.items()}
        nx.draw_networkx_labels(self.graph, pos=pos_higher, font_size=5, font_color="black", ax=ax)

        edge_labels = {(u, v): index for index, (u, v, _) in enumerate(self.graph.edges(data=True))}
        nx.draw_networkx_edge_labels(self.graph, pos=pos, font_size=4, edge_labels=edge_labels, font_color="black", ax=ax)

        # Return the figure
        return fig


    # --------------------- IERS
    # TODO: make more of the iers more custom (e.g. number, protocol, load)
    def generate_iers(self, count: int = 5) -> None:
        for i in range(count):
            edge = random.choice(list(self.links_dict.values()))
            ier_type = random.choice(list(IERType))
            start_step = random.randint(1, 127)
            end_step = random.randint(start_step, 128)
            service = random.choice(self.services)
            port = SERVICE_TO_PORT_MAP[service][0]

            self.add_ier(
                _type=ier_type,
                _id=str(i),
                _start_step=start_step,
                _end_step=end_step,
                _load=10000,
                _protocol=service,
                _port=port,
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
        
    def generate_laydown(self, laydown_save_path: str | Path | None) -> None:
        self.laydown = Laydown(
            initial_network=self.network,
            iers=self.iers,
            red_pols=self.red_pols,
            green_pols=self.green_pols
        )

        if laydown_save_path:
            self.laydown_save_path = laydown_save_path
            self.laydown.save(laydown_save_path)
    
    # --------------------- POLS
    def generate_green_pols(self, count: int = 5) -> None:
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
                _service_name=random.choice(self.services),
                _state=state
            )

    def add_green_pol(self, **kwargs) -> None:
        pol = NodeStateInstructionGreen(**kwargs)
        self.green_pols.append(pol)

    def generate_red_pols(self, count: int=5) -> None:
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

            service = random.choice(self.services)

            self.add_red_pol(
                _id=str(i),
                _start_step=start_step,
                _end_step=random.randint(start_step, 128),
                _target_node_id=edge.dest_node_id,
                _pol_initiator=NodePOLInitiator.IER,
                _pol_type=pol_type,
                pol_protocol=service,
                _pol_state=pol_state,
                _pol_source_node_id=edge.source_node_id,
                _pol_source_node_service=service,
                _pol_source_node_service_state=pol_state
            )

    def add_red_pol(self, **kwargs) -> None:
        pol = NodeStateInstructionRed(**kwargs)
        self.red_pols.append(pol)


    # --------------------- TARGET STUFF

    def generate_target_action(self) -> None:
        reasoning = self.pretrained_llm.generate_prompt_response(
            prompt_template=REASON_ACTION_SPACE_NODE_SELECT_TEMPLATE,
            grammar=AgentReasoningNodeSelection,
            network_description=self.network_description,
            node_list=list(self.nodes_dict.values()),
            action_info=ACTION_INFO.format(service_names=self.services)
        )

        action = self.pretrained_llm.generate_prompt_response(
            prompt_template=ACTION_SELECTION_TEMPLATE,
            grammar=AgentNodeAction,
            network_description=self.network_description,
            reasoning_statement=reasoning.reasoning,
            action_info=ACTION_INFO.format(service_names=self.services)
        )

        self.target_reasoning = reasoning
        self.target_action = action

    def create_simulation(self) -> None:

        if not self.laydown_save_path:
            raise Exception("Need a laydown. Please use the `generate_laydown` first")

        self.simulation = Simulation.from_laydown(laydown_path=self.laydown_save_path, training_path=self.training_config_path)
        self.graph_as_red_and_blue = self.simulation.latest_state.network_figure

    def create_output_graph_data(self) -> Data:
        obs = self.simulation.env.reset()
        _network = self.simulation.env.network

        graph = prepare_graph(_network)
        state = from_networkx(graph)
        state.x = torch.tensor(obs[:_network.number_of_nodes(), 1:], dtype=torch.float32).to('cpu')
        return state
    def get_prompt_data(self):
        node_mapping = self.simulation.env.nodes
        service_mapping = {key + 1: value for key, value in enumerate(self.simulation.env.services_list)}
        
        node_services = {}
        for node in self.simulation.env.nodes.values():
            node_services[node.name] = []
            for service in self.simulation.env.services_list:
                # Apparently only these NodeUnion types can have services?
                if node.node_type == NodeType.SERVER or node.node_type == NodeType.COMPUTER:
                    if node.has_service(service):
                        node_services[node.name].append([k for k, v in service_mapping.items() if v == service][0])
        
        
        # Get stringified node map, service map, node services
        node_str = "\n".join(f"{key}: {value}" for key, value in node_mapping.items())  # ID: NODE_NAME
        services_str = "\n".join(f"{i+ 1}: {service_mapping[i+1]}" for i, _ in enumerate(service_mapping))
        node_services_str = "\n".join(
            f"{key}: {', '.join(str(val) for val in value) if value else 'NONE'}"
            for key, value in node_services.items()
        )
        
        return node_str, services_str, node_services_str
        
    def run_end_to_end(self, laydown_save_path: str | Path | None) -> Data:
        # add iers
        self.generate_iers()
        self.generate_green_pols()
        self.generate_red_pols()

        # create the laydown
        self.generate_laydown(laydown_save_path)

        # init primate session
        session = PrimaiteSession(training_config_path=self.training_config_path, lay_down_config_path=laydown_save_path)
        session.setup()

        # generate target action stuff
        self.generate_target_action()

        # simulation
        self.create_simulation()

        # Create prompt data for GLLM
        node_str, services_str, node_services_str = self.get_prompt_data()
        output_graph_data = self.create_output_graph_data()
        # add all the useful stuff to the graph output 🤢
        output_graph_data.reasoning = self.target_reasoning.reasoning
        output_graph_data.chosen_node = self.target_action.node_name
        output_graph_data.node_property = self.target_action.node_property
        output_graph_data.property_action = self.target_action.property_action
        output_graph_data.service_name = self.target_action.service_name
        output_graph_data.laydown_filename = os.path.basename(laydown_save_path)
        output_graph_data.node_ids = node_str
        output_graph_data.services = services_str
        output_graph_data.node_services = node_services_str

        return output_graph_data


