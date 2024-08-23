from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import cached_property

import matplotlib.pyplot as plt
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

    def display_graph(self):
        # Make sure node locations in plot are constant
        G = self.graph
        pos = nx.spring_layout(G, seed=100)
        fig = plt.figure()

        # Draw nodes
        # Nodes which have at least one of the states not being NONE or ON/GOOD are marked as RED
        node_color_map = []
        for G_node in G:
            node_id = G_node.node_id
            node = self.nodes_dict[node_id]
            if node.is_working():
                node_color_map.append("tab:blue")
            else:
                node_color_map.append("tab:red")

        # Draw edges
        # The higher the traffic in an edge, the more red the link is
        edge_color_map = []
        for src, dest, data in G.edges(data=True):
            link_id = data["id"]
            link = self.links_dict[link_id]
            traffic_level = link.get_traffic_level()
            edge_color_map.append(traffic_level)

        nx.draw_networkx(
            G,
            pos=pos,
            node_color=node_color_map,
            with_labels=False,
            edge_color=edge_color_map,
            edge_cmap=plt.cm.hot,  # type: ignore
            edge_vmax=1.6,
        )

        pos_higher = {}
        y_off = 0.05  # offset on the y axis

        for k, v in pos.items():
            pos_higher[k] = (v[0], v[1] + y_off)

        nx.draw_networkx_labels(G, pos=pos_higher, font_size=5, font_color="black")

        edge_labels = {edge[0:2]: edge[2]["id"] for edge in G.edges(data=True)}

        nx.draw_networkx_edge_labels(G, pos=pos, font_size=4, edge_labels=edge_labels, font_color="black")

        return fig
    
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
