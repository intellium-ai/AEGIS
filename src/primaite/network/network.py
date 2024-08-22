from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import cached_property

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import networkx as nx
import pandas as pd

from primaite.common.custom_typing import Serializable
from primaite.common.enums import SoftwareState
from primaite.environment.primaite_env import Primaite
from primaite.links import Link
from primaite.nodes import ActiveNode, Node, NodeUnion, ServiceNode
from primaite.common.enums import NodeType, SoftwareState



@dataclass
class Network(Serializable):
    """A network frozen in a particular state"""

    nodes_dict: dict[str, NodeUnion]
    links_dict: dict[str, Link]
    service_names: list[str]
    ports_list: list[str]
    graph: nx.Graph

    @classmethod
    def from_env(cls, env: Primaite):
        nodes_dict = deepcopy(env.nodes)
        links_dict = deepcopy(env.links)
        graph = env.network.copy()
        service_names = env.services_list
        ports_list = env.ports_list

        return cls(nodes_dict, links_dict, service_names, ports_list, graph)

    @cached_property
    def nodes(self) -> list[Node]:
        return list(self.nodes_dict.values())

    @cached_property
    def active_nodes(self) -> list[ActiveNode]:
        return [n for n in self.nodes if isinstance(n, ActiveNode)]

    @cached_property
    def service_nodes(self) -> list[ServiceNode]:
        return [n for n in self.active_nodes if isinstance(n, ServiceNode)]

    @cached_property
    def links(self) -> list[Link]:
        return list(self.links_dict.values())

    def nodes_table(self, limited_view=False) -> pd.DataFrame:
        nodes = self.active_nodes

        nodes_dict = {
            "Name": [n.name for n in nodes],
            # "Type": [n.node_type.name for n in nodes],
            # "IP": [n.ip_address for n in nodes],
            "Hardware State": [n.hardware_state.name for n in nodes],
            "Software State": [n.software_state.name for n in nodes],
            "File System State": [
                n.file_system_state_observed.name if limited_view else n.file_system_state_actual.name for n in nodes
            ],
        }

        services_dict = {
            protocol
            + " Service State": [
                node.get_service_state(protocol).name if isinstance(node, ServiceNode) else SoftwareState.NONE.name
                for node in nodes
            ]
            for protocol in self.service_names
        }

        nodes_table = pd.DataFrame(nodes_dict | services_dict)
        nodes_table.index += 1
        nodes_table.replace({"NONE": "-"}, inplace=True)

        return nodes_table

    def traffic_table(self) -> pd.DataFrame:

        indices = [i for i in range(len(self.links))]

        traffic_dict = {
            protocol
            + " Traffic": [int(100 * link.get_current_protocol_load(protocol) / link.bandwidth) for link in self.links]
            for protocol in self.service_names
        }
        link_dict = {"Name": list(self.links_dict.keys())}

        traffic_table = pd.DataFrame(link_dict | traffic_dict)
        traffic_table.index = indices  # type: ignore

        return traffic_table

    def display_graph(self, random_seed: int = 1729) -> None:

        # NOTE: not sure the color of links changes with traffic

        pos = nx.spring_layout(self.graph, seed=random_seed)

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
        for index, (src_node, dest_node, data) in enumerate(self.graph.edges(data=True)):
            link_id = f"link_{index}"
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

        plt.show()

    def diff(self, prev: Network, limited_view=False, colors=True) -> list[str]:
        diff = []
        curr = self
        # Nodes

        curr_nodes_table = curr.nodes_table(limited_view=limited_view)
        prev_nodes_table = prev.nodes_table(limited_view=limited_view)
        compare_nodes = prev_nodes_table.compare(curr_nodes_table, result_names=("prev", "curr"), align_axis=0)
        for state_name in compare_nodes:
            for id, s in compare_nodes[state_name].groupby(level=0):
                node_name = curr_nodes_table.at[id, "Name"]
                prev_state = s.iloc[0]
                curr_state = s.iloc[1]
                if colors:
                    node_change_str = f"Node :blue[{node_name}]'s :violet[{state_name}] changed from :red[{prev_state}] to :red[{curr_state}]."
                else:
                    node_change_str = f"Node {node_name}'s {state_name} changed from {prev_state} to {curr_state}."
                diff.append(node_change_str)

        # Links
        compare_traffic = prev.traffic_table().compare(
            curr.traffic_table(), result_names=("prev", "curr"), align_axis=0
        )
        for service_name in compare_traffic:
            for id, s in compare_traffic[service_name].groupby(level=0):
                link_name = curr.traffic_table().at[id, "Name"]
                prev_traffic = s.iloc[0]
                curr_traffic = s.iloc[1]
                traffic_diff = int(curr_traffic - prev_traffic)
                if colors:
                    link_change_str = (
                        f":violet[{service_name}] in link :green[{link_name}] changed by :red[{traffic_diff}]."
                    )
                else:
                    link_change_str = f"{service_name} in link {link_name} changed by {traffic_diff}."
                diff.append(link_change_str)

        return diff

    def serialize(self):
        ports = {"item_type": "PORTS", "ports_list": [{"port": f"{port}"} for port in self.ports_list]}
        services = {"item_type": "SERVICES", "service_list": [{"name": f"{service}"} for service in self.service_names]}
        nodes = [node.serialize() for node in self.nodes]
        links = [link.serialize() for link in self.links]

        return [ports, services, *nodes, *links]
