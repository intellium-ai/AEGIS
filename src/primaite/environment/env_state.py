# Displaying the environment state
from __future__ import annotations

import copy
from dataclasses import dataclass

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from primaite.common.enums import FileSystemState, HardwareState, NodeType, RulePermissionType, SoftwareState
from primaite.environment.primaite_env import Primaite
from primaite.links.link import Link
from primaite.nodes.active_node import ActiveNode
from primaite.nodes.node import Node
from primaite.nodes.service_node import ServiceNode


class EnvironmentState:

    def __init__(
        self,
        env: Primaite,
        prev_env_state: EnvironmentState | None = None,
        action_id: int | None = None,
        prompt: str | None = None,
        reasoning: str | None = None,
    ):

        self.prompt = prompt
        self.reasoning = reasoning
        self.action_id = action_id
        self.env = env
        self.nodes = copy.deepcopy(env.nodes)
        self.links = copy.deepcopy(env.links)
        self.prev_env_state = prev_env_state
        self.nodes_table = get_nodes_table(env)
        self.traffic_table = get_traffic_table(env)
        self.acl_table = get_acl_table(env)
        self.network = env.network.copy()

    def obs_diff(self, colors=True) -> list[str]:
        if self.prev_env_state is None:
            return []

        diff = []
        # Nodes
        prev_nodes_table = self.prev_env_state.nodes_table
        compare_nodes = prev_nodes_table.compare(self.nodes_table, result_names=("prev", "curr"), align_axis=0)
        for state_name in compare_nodes:
            for id, s in compare_nodes[state_name].groupby(level=0):
                node_name = self.nodes_table.at[id, "Name"]
                prev_state = s.iloc[0]
                curr_state = s.iloc[1]
                if colors:
                    node_change_str = f"Node :blue[{node_name}]'s :violet[{state_name}] changed from :red[{prev_state}] to :red[{curr_state}]."
                else:
                    node_change_str = f"Node {node_name}'s {state_name} changed from {prev_state} to {curr_state}."
                diff.append(node_change_str)

        # Links
        prev_traffic_table = self.prev_env_state.traffic_table
        compare_traffic = prev_traffic_table.compare(self.traffic_table, result_names=("prev", "curr"), align_axis=0)
        for service_name in compare_traffic:
            for id, s in compare_traffic[service_name].groupby(level=0):
                link_name = self.traffic_table.at[id, "Name"]
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

    def display_network(self):
        # Make sure node locations in plot are constant
        G = self.network
        pos = nx.spring_layout(G, seed=100)
        fig = plt.figure()

        # Draw nodes
        # Nodes which have at least one of the states not being NONE or ON/GOOD are marked as RED
        node_color_map = []
        for G_node in G:
            node_id = G_node.node_id
            node = self.nodes[node_id]
            if _is_working(node):
                node_color_map.append("tab:blue")
            else:
                node_color_map.append("tab:red")

        # Draw edges
        # The higher the traffic in an edge, the more red the link is
        edge_color_map = []
        for src, dest, data in G.edges(data=True):
            link_id = data["id"]
            link = self.links[link_id]
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


def get_acl_table(env: Primaite) -> pd.DataFrame:
    rules = env.acl.acl

    rules = [r for r in rules if r is not None]

    rules_dict = {
        "Permission": ["ALLOW" if r.permission == RulePermissionType.ALLOW else "DENY" for r in rules],
        "Source IP": [r.source_ip for r in rules],
        "Dest IP": [r.dest_ip for r in rules],
        "Protocol": [r.protocol for r in rules],
        "Port": [r.port for r in rules],
    }

    acl_table = pd.DataFrame(rules_dict, index=None)

    return acl_table


def get_traffic_table(env: Primaite) -> pd.DataFrame:
    links = list(env.links.values())

    indices = [int(link.id) for link in links]

    protocols = env.services_list
    traffic_dict = {
        protocol + " Traffic": [int(100 * link.get_current_protocol_load(protocol) / link.bandwidth) for link in links]
        for protocol in protocols
    }
    link_dict = {"Name": list(env.links.keys())}

    traffic_table = pd.DataFrame(link_dict | traffic_dict)
    traffic_table.index = indices  # type: ignore

    return traffic_table


def get_nodes_table(env: Primaite) -> pd.DataFrame:
    nodes = env.active_nodes

    nodes_dict = {
        "Name": [n.name for n in nodes],
        # "Type": [n.node_type.name for n in nodes],
        # "IP": [n.ip_address for n in nodes],
        "Hardware State": [n.hardware_state.name for n in nodes],
        "Software State": [n.software_state.name for n in nodes],
        "File System State": [n.file_system_state_observed.name for n in nodes],
    }

    protocols = env.services_list
    services_dict = {
        protocol
        + " Service State": [
            node.get_service_state(protocol).name if isinstance(node, ServiceNode) else SoftwareState.NONE.name
            for node in nodes
        ]
        for protocol in protocols
    }

    nodes_table = pd.DataFrame(nodes_dict | services_dict)
    nodes_table.index += 1
    nodes_table.replace({"NONE": "-"}, inplace=True)

    return nodes_table


def _is_working(node: Node) -> bool:
    if node.hardware_state.value > 1:
        return False
    if isinstance(node, ActiveNode):
        if node.software_state.value > 1:
            return False
        if node.file_system_state_observed.value > 1:
            return False

    if isinstance(node, ServiceNode):
        for s in list(node.services.values()):
            if s.software_state.value > 1:
                return False

    return True
