### (Jack) ### This file was mostly for me to muck around with obs and action spaces.
### Polished functions can be found in primaite/agents/utils.py

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from primaite.action import NodeAction
from primaite.environment.primaite_env import Primaite

HARDWARE_STATE: Dict[int, str] = {0: "none", 1: "on", 2: "off", 3: "resetting", 4: "shutting down", 5: "booting"}
SOFTWARE_STATE: Dict[int, str] = {0: "none", 1: "good", 2: "patching", 3: "compromised", 4: "overwhelmed"}
FILESYSTEM_STATE: Dict[int, str] = {0: "none", 1: "good", 2: "corrupt", 3: "destroyed", 4: "repairing", 5: "restoring"}
SERVICE_STATE: Dict[int, str] = {0: "none", 1: "good", 2: "patching", 3: "compromised", 4: "overwhelmed"}
TRAFFIC_LEVEL: Dict[int, str] = {
    0: "no traffic",
    1: "low traffic",
    2: "medium traffic",
    3: "high traffic",
    4: "overwhelmed",
}


def init_labels(num_services) -> List[str]:
    labels = ["hardware status", "software status", "filesystem status"]
    for i in range(num_services):
        labels.append(f"service {i+1} status")
    return labels


def init_node_state_dict(num_services: int) -> List[Dict[int, str]]:
    state = [HARDWARE_STATE, SOFTWARE_STATE, FILESYSTEM_STATE]
    state.extend([SERVICE_STATE for _ in range(num_services)])
    return state


def linktraffic_to_understandable(obs: np.ndarray, num_links: int) -> Dict[str, str]:
    assert len(obs) == num_links
    readable_traffic = {}
    for idx, link in enumerate(obs):
        readable_traffic[f"link {idx + 1}"] = TRAFFIC_LEVEL[link]
    return readable_traffic


def nodestatus_to_understandable(obs: np.ndarray, num_nodes: int, num_services: int) -> Dict[str, Any]:
    node_obs = np.split(obs, num_nodes)
    readable_network_obs = {}
    for node_idx, node in enumerate(node_obs):
        readable_state = {}
        labels = init_labels(num_services=num_services)
        node_state_dict = init_node_state_dict(num_services=num_services)
        for idx, component in enumerate(node_state_dict):
            readable_state[labels[idx]] = component[node[idx]]
        readable_network_obs[f"node {node_idx + 1} status"] = readable_state
    return readable_network_obs


def nodelink_to_understandable(
    obs: np.ndarray,
    num_nodes: int,
    num_services: int,
    num_links: int,
) -> Tuple[List[List[str]], List[List[int]]]:
    assert len(obs) == num_nodes + num_links

    nodes = obs[:num_nodes]
    links = obs[num_nodes:]

    readable_node_obs = []
    for node in nodes:
        readable_state = []
        node_state_dict = init_node_state_dict(num_services)
        for idx, component in enumerate(node_state_dict):
            readable_state.append(component[node[idx + 1]])
        readable_node_obs.append(readable_state)

    readable_link_obs = []
    for link in links:
        readable_link_obs.append(list(link[num_services + 1 :]))

    return readable_node_obs, readable_link_obs
