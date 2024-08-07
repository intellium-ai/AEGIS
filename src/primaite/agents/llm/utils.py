### (Jack) ### This file was mostly for me to muck around with obs and action spaces.
### Polished functions can be found in primaite/agents/utils.py

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from primaite.environment.env_state import EnvironmentState
from primaite.environment.primaite_env import Primaite
from primaite.action import NodeAction

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


def network_connectivity_desc(env_state: EnvironmentState) -> str:
    # List nodes
    desc = ""
    nodes = list(env_state.nodes.values())
    nodes_df = pd.DataFrame({"Name": [n.name for n in nodes], "Type": [n.node_type.name for n in nodes]})
    desc += "\nNodes:\n" + nodes_df.to_json(orient="records", indent=2)
    links = list(env_state.links.values())
    links_df = pd.DataFrame(
        {
            "Name": list(env_state.links.keys()),
            "Source Node": [l.source_node_name for l in links],
            "Destination Node": [l.dest_node_name for l in links],
        }
    )
    desc += "\n\nLinks:\n" + links_df.to_json(orient="records", indent=2)

    return desc


def obs_view_full(env_state: EnvironmentState) -> str:
    obs_str = ""
    nodes = env_state.nodes_table.to_json(orient="records", indent=2)
    obs_str += "\n\nNode Status:\n" + nodes
    links = env_state.traffic_table.to_json(orient="records", indent=2)
    obs_str += "\n\nTraffic Status:\n" + links + "\n\n"

    return obs_str


def obs_diff(env_state: EnvironmentState) -> str:
    obs_diff = env_state.obs_diff(colors=False)
    if obs_diff:
        obs_str = "\n".join(obs_diff)
    else:
        obs_str = "NO CHANGE"
    return obs_str


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


def get_obs_act_history_str(env_history: List[EnvironmentState], env: Primaite, max_history: int = 20) -> str:
    """Builds the observation history string used in LLM agent prompts. Uses the latest N in the history"""
    # Build the history of actions
    obs_act_history = "" if env_history else "NO OBSERVATION HISTORY"
    history_list = []

    for i, state in enumerate(env_history[1:]):
        observed_changes = obs_diff(state)
        action_id = state.action_id

        if observed_changes != "" or action_id:
            history_list.append(f"\nStep {i}:")

            if observed_changes != "":
                history_list[-1] += f"\n{observed_changes}"

            if action_id is not None:
                action = NodeAction.from_id(env=env, action_id=action_id)
                action_verbose = action.verbose(colored=False)
                history_list[-1] += f"\nAction: {action_verbose}\n"

    obs_act_history += "".join(
        history_list[-max_history:]
    )  # Only show up to the last 20 obs act events to avoid cloggage

    return obs_act_history
