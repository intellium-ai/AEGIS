# © Crown-owned copyright 2023, Defence Science and Technology Laboratory UK
from collections import defaultdict
from typing import Any, Dict, List, Literal, Optional, Union

import networkx as nx
import numpy as np
import torch
import torch_geometric
import torch_geometric.data
from torch_geometric.data import Data

from primaite import getLogger
from primaite.common.enums import (
    FileSystemState,
    HardwareState,
    LinkStatus,
    NodeFileSystemAction,
    NodeHardwareAction,
    NodePOLType,
    NodeSoftwareAction,
    RulePermissionType,
    SoftwareState,
)
from primaite.nodes import NodeUnion

_LOGGER = getLogger(__name__)


def transform_action_node_readable(action: List[int]) -> List[Union[int, str]]:
    """Convert a node action from enumerated format to readable format.

    example:
    [1, 3, 1, 0] -> [1, 'SERVICE', 'PATCHING', 0]

    :param action: Agent action, formatted as a list of ints, for more information check out
        `primaite.environment.primaite_env.Primaite`
    :type action: List[int]
    :return: The same action list, but with the encodings translated back into meaningful labels
    :rtype: List[Union[int,str]]
    """
    action_node_property = NodePOLType(action[1]).name

    if action_node_property == "OPERATING":
        property_action = NodeHardwareAction(action[2]).name
    elif (action_node_property == "OS" or action_node_property == "SERVICE") and action[2] <= 1:
        property_action = NodeSoftwareAction(action[2]).name
    elif action_node_property == "FILE":
        property_action = NodeFileSystemAction(action[2]).name
    else:
        property_action = "NONE"

    new_action: list[Union[int, str]] = [action[0], action_node_property, property_action, action[3]]
    return new_action


def transform_action_acl_readable(action: List[int]) -> List[Union[str, int]]:
    """
    Transform an ACL action to a more readable format.

    example:
    [0, 1, 2, 5, 0, 1] -> ['NONE', 'ALLOW', 2, 5, 'ANY', 1]

    :param action: Agent action, formatted as a list of ints, for more information check out
        `primaite.environment.primaite_env.Primaite`
    :type action: List[int]
    :return: The same action list, but with the encodings translated back into meaningful labels
    :rtype: List[Union[int,str]]
    """
    action_decisions = {0: "NONE", 1: "CREATE", 2: "DELETE"}
    action_permissions = {0: "DENY", 1: "ALLOW"}

    action_decision = action_decisions[action[0]]
    action_permission = action_permissions[action[1]]

    # For IPs, Ports and Protocols, 0 means any, otherwise its just an index
    new_action = [action_decision, action_permission] + list(action[2:6])
    for n, val in enumerate(list(action[2:6])):
        if val == 0:
            new_action[n + 2] = "ANY"

    return new_action


def _transform_change_nodelink_readable(obs: np.ndarray) -> List[List[Union[str, int]]]:
    """Transform nodelink table readable list of each observation property.

    example:
    np.array([[1,2,1,3],[2,1,1,1]]) -> [[1, 2], ['OFF', 'ON'], ['GOOD', 'GOOD'], ['COMPROMISED', 'GOOD']]

    :param obs: Raw observation from the environment.
    :type obs: np.ndarray
    :return: The same observation, but the encoded integer values are replaced with readable names.
    :rtype: List[List[Union[str, int]]]
    """
    ids = [i for i in obs[:, 0]]
    operating_states = [HardwareState(i).name for i in obs[:, 1]]
    os_states = [SoftwareState(i).name for i in obs[:, 2]]
    fs_states = [FileSystemState(i).name for i in obs[:, 3]]
    new_obs = [ids, operating_states, os_states, fs_states]

    for service in range(4, obs.shape[1]):
        # Links bit/s don't have a service state
        service_states = [SoftwareState(i).name if i <= 4 else i for i in obs[:, service]]
        new_obs.append(service_states)

    return new_obs


def _transform_change_nodestatus_readable(obs: np.ndarray, num_nodes: int) -> List[List[str]]:
    """
    Transform NodeStatus observation space into readable array.

    Args:
        obs (np.ndarray): Array of nodestatus observation space
        num_nodes (int): Number of nodes in the array

    Returns:
        List[List[str]]: Readable version of observation space. See docs for details
    """
    nodes = np.array(np.split(obs, num_nodes))

    ids = list(range(1, num_nodes + 1))
    operating_states = [HardwareState(i).name for i in nodes[:, 0]]
    os_states = [SoftwareState(i).name for i in nodes[:, 1]]
    fs_states = [FileSystemState(i).name for i in nodes[:, 2]]

    new_obs = [ids, operating_states, os_states, fs_states]

    for col in range(3, nodes.shape[1]):
        new_obs.append([SoftwareState(i).name for i in nodes[:, col]])

    return new_obs


def transform_nodelink_readable(obs: np.ndarray) -> List[List[Union[str, int]]]:
    """
    Reformat nodelink table to be transposed, i.e. List of nodes for each property, to list of properties for each node.

    Args:
        obs (np.ndarray): _description_
        num_nodes (int): _description_

    Returns:
        List[List[str]]: _description_
    """
    changed_obs = _transform_change_nodelink_readable(obs)
    new_obs = list(zip(*changed_obs))
    # Convert list of tuples to list of lists
    new_obs = [list(i) for i in new_obs]

    return new_obs


def transform_nodestatus_readable(obs: np.ndarray, num_nodes: int) -> List[List[str]]:
    """
    Reformat nodestatus table to be transposed, i.e. List of nodes for each property, to list of properties for each node.

    Args:
        obs (np.ndarray): _description_
        num_nodes (int): _description_

    Returns:
        List[List[str]]: _description_
    """
    changed_obs = _transform_change_nodestatus_readable(obs=obs, num_nodes=num_nodes)
    new_obs = list(zip(*changed_obs))
    return [list(i) for i in new_obs]


def convert_to_new_obs(obs: np.ndarray, num_nodes: int = 10) -> np.ndarray:
    """Convert original gym Box observation space to new multiDiscrete observation space.

    :param obs: observation in the 'old' (NodeLinkTable) format
    :type obs: np.ndarray
    :param num_nodes: number of nodes in the network, defaults to 10
    :type num_nodes: int, optional
    :return: reformatted observation
    :rtype: np.ndarray
    """
    # Remove ID columns, remove links and flatten to MultiDiscrete observation space
    new_obs = obs[:num_nodes, 1:].flatten()
    return new_obs


def convert_to_old_obs(obs: np.ndarray, num_nodes: int = 10, num_links: int = 10, num_services: int = 1) -> np.ndarray:
    """Convert to old observation.

    Links filled with 0's as no information is included in new observation space.

    example:
    obs = array([1, 1, 1, 1, 1, 1, 1, 1, 1,  ..., 1, 1, 1])

    new_obs = array([[ 1,  1,  1,  1],
                     [ 2,  1,  1,  1],
                     [ 3,  1,  1,  1],
                     ...
                    [20,  0,  0,  0]])

    :param obs: observation in the 'new' (MultiDiscrete) format
    :type obs: np.ndarray
    :param num_nodes: number of nodes in the network, defaults to 10
    :type num_nodes: int, optional
    :param num_links: number of links in the network, defaults to 10
    :type num_links: int, optional
    :param num_services: number of services on the network, defaults to 1
    :type num_services: int, optional
    :return: 2-d BOX observation space, in the same format as NodeLinkTable
    :rtype: np.ndarray
    """
    # Convert back to more readable, original format
    reshaped_nodes = obs[:-num_links].reshape(num_nodes, num_services + 2)

    # Add empty links back and add node ID back
    s = np.zeros(
        [reshaped_nodes.shape[0] + num_links, reshaped_nodes.shape[1] + 1],
        dtype=np.int64,
    )
    s[:, 0] = range(1, num_nodes + num_links + 1)  # Adding ID back
    s[:num_nodes, 1:] = reshaped_nodes  # put values back in
    new_obs = s

    # Add links back in
    links = obs[-num_links:]
    # Links will be added to the last protocol/service slot but they are not specific to that service
    new_obs[num_nodes:, -1] = links

    return new_obs


def describe_obs_change(
    obs1: np.ndarray, obs2: np.ndarray, num_nodes: int = 10, num_links: int = 10, num_services: int = 1
) -> str:
    """Build a string describing the difference between two observations.

    example:
    obs_1 = array([[1, 1, 1, 1, 3], [2, 1, 1, 1, 1]])
    obs_2 = array([[1, 1, 1, 1, 1], [2, 1, 1, 1, 1]])
    output = 'ID 1: SERVICE 2 set to GOOD'

    :param obs1: First observation
    :type obs1: np.ndarray
    :param obs2: Second observation
    :type obs2: np.ndarray
    :param num_nodes: How many nodes are in the network laydown, defaults to 10
    :type num_nodes: int, optional
    :param num_links: How many links are in the network laydown, defaults to 10
    :type num_links: int, optional
    :param num_services: How many services are configured for this scenario, defaults to 1
    :type num_services: int, optional
    :return: A multi-line string with a human-readable description of the difference.
    :rtype: str
    """
    # obs1 = convert_to_old_obs(obs1, num_nodes, num_links, num_services)
    # obs2 = convert_to_old_obs(obs2, num_nodes, num_links, num_services)
    list_of_changes = []
    for n, row in enumerate(obs1 - obs2):
        if row.any() != 0:
            relevant_changes = np.where(row != 0, obs2[n], -1)
            relevant_changes[0] = obs2[n, 0]  # ID is always relevant
            is_link = relevant_changes[0] > num_nodes
            desc = _describe_obs_change_helper(list(relevant_changes), is_link)
            list_of_changes.append(desc)

    change_string = "\n ".join(list_of_changes)
    if len(list_of_changes) > 0:
        change_string = "\n " + change_string
    return change_string


def _describe_obs_change_helper(obs_change: List[int], is_link: bool) -> str:
    """
    Helper funcion to describe what has changed.

    example:
    [ 1 -1 -1 -1  1] -> "ID 1: Service 1 changed to GOOD"

    Handles multiple changes e.g. 'ID 1: SERVICE 1 changed to PATCHING. SERVICE 2 set to GOOD.'

    :param obs_change: List of integers generated within the `describe_obs_change` function. It should correspond to one
        row of the observation table, and have `-1` at locations where the observation hasn't changed, and the new
        status where it has changed.
    :type obs_change: List[int]
    :param is_link: Whether the row of the observation space corresponds to a link. False means it represents a node.
    :type is_link: bool
    :return: A human-readable description of the difference between the two observation rows.
    :rtype: str
    """
    # Indexes where a change has occured, not including 0th index
    index_changed = [i for i in range(1, len(obs_change)) if obs_change[i] != -1]
    # Node pol types, Indexes >= 3 are service nodes
    NodePOLTypes = [NodePOLType(i).name if i < 3 else NodePOLType(3).name + " " + str(i - 3) for i in index_changed]
    # Account for hardware states, software sattes and links
    states = [
        (
            LinkStatus(obs_change[i]).name
            if is_link
            else HardwareState(obs_change[i]).name if i == 1 else SoftwareState(obs_change[i]).name
        )
        for i in index_changed
    ]

    if not is_link:
        desc = f"ID {obs_change[0]}:"
        for node_pol_type, state in list(zip(NodePOLTypes, states)):
            desc = desc + " " + node_pol_type + " changed to " + state + "."
    else:
        desc = f"ID {obs_change[0]}: Link traffic changed to {states[0]}."

    return desc


def transform_action_node_enum(action: List[Union[str, int]]) -> List[int]:
    """Convert a node action from readable string format, to enumerated format.

    example:
    [1, 'SERVICE', 'PATCHING', 0] -> [1, 3, 1, 0]
    :param action: Action in 'readable' format
    :type action: List[Union[str,int]]
    :return: Action with verbs encoded as ints
    :rtype: List[int]
    """
    action_node_id = action[0]
    action_node_property = NodePOLType[str(action[1])].value

    if action[1] == "OPERATING":
        property_action = NodeHardwareAction[str(action[2])].value
    elif action[1] == "OS" or action[1] == "SERVICE":
        property_action = NodeSoftwareAction[
            str(action[2])
        ].value  # added if because for some reason they made this enum patch but pass patching for hardcoded agent only...

    else:
        property_action = 0

    action_service_index = action[3]

    new_action = [
        action_node_id,
        action_node_property,
        property_action,
        action_service_index,
    ]

    return new_action


def transform_action_acl_enum(action: List[Union[int, str]]) -> np.ndarray:
    """
    Convert acl action from readable str format, to enumerated format.

    :param action: ACL-based action expressed as a list of human-readable ints and strings
    :type action: List[Union[int,str]]
    :return: The same action but encoded to contain only integers.
    :rtype: np.ndarray
    """
    action_decisions = {"NONE": 0, "CREATE": 1, "DELETE": 2}
    action_permissions = {"DENY": 0, "ALLOW": 1}

    action_decision = action_decisions[str(action[0])]
    action_permission = action_permissions[str(action[1])]

    # For IPs, Ports and Protocols, ANY has value 0, otherwise its just an index
    new_action = [action_decision, action_permission] + list(action[2:6])
    for n, val in enumerate(list(action[2:6])):
        if val == "ANY":
            new_action[n + 2] = 0

    new_action = np.array(new_action)
    return new_action


def get_node_of_ip(ip: str, node_dict: Dict[str, NodeUnion]) -> str:
    """Get the node ID of an IP address.

    node_dict: dictionary of nodes where key is ID, and value is the node (can be ontained from env.nodes)

    :param ip: The IP address of the node whose ID is required
    :type ip: str
    :param node_dict: The environment's node registry dictionary
    :type node_dict: Dict[str,NodeUnion]
    :return: The key from the registry dict that corresponds to the node with the IP adress provided by `ip`
    :rtype: str
    """
    for node_key, node_value in node_dict.items():
        node_ip = node_value.ip_address
        if node_ip == ip:
            return node_key
    return ""


def get_new_action(old_action: np.ndarray, action_dict: Dict[int, List]) -> int:
    """
    Get new action (e.g. 32) from old action e.g. [1,1,1,0].

    Old_action can be either node or acl action type

    :param old_action: Action expressed as a list of choices, eg. [1,1,1,0]
    :type old_action: np.ndarray
    :param action_dict: Dictionary for translating the multidiscrete actions into the list-based actions.
    :type action_dict: Dict[int,List]
    :return: Action key correspoinding to the input `old_action`
    :rtype: int
    """
    for key, val in action_dict.items():
        if list(val) == list(old_action):
            return key
    # Not all possible actions are included in dict, only valid action are
    # if action is not in the dict, its an invalid action so return 0
    return 0


def split_obs_space(obs: np.ndarray, num_nodes: int, num_links: int, num_services: int):
    """
    WIP: When multiple observation spaces are specified in the config file, they are flattened into a 1D array.
    This function should separate them out into their original obs spaces in the correct shape.

    Args:
        obs (np.ndarray): Observation array
        num_nodes (int): Number of nodes in the network
        num_links (int): Number of links in the network
        num_services (int): Number of services in the network

    Returns:
        Tuple: Individual obs spaces
    """
    nodelink_size = (num_nodes + num_links) * (num_services + 4)
    return obs[:nodelink_size]


def from_networkx(
    G: Any,
    group_node_attrs: Optional[Union[List[str], Literal["all"]]] = None,
    group_edge_attrs: Optional[Union[List[str], Literal["all"]]] = None,
) -> torch_geometric.data.Data:
    r"""
    (Jack) I stole this from torch geo as i needed to make some slight modifications

    Converts a :obj:`networkx.Graph` or :obj:`networkx.DiGraph` to a
    :class:`torch_geometric.data.Data` instance.

    Args:
        G (networkx.Graph or networkx.DiGraph): A networkx graph.
        group_node_attrs (List[str] or "all", optional): The node attributes to
            be concatenated and added to :obj:`data.x`. (default: :obj:`None`)
        group_edge_attrs (List[str] or "all", optional): The edge attributes to
            be concatenated and added to :obj:`data.edge_attr`.
            (default: :obj:`None`)

    .. note::

        All :attr:`group_node_attrs` and :attr:`group_edge_attrs` values must
        be numeric.

    Examples:
        >>> edge_index = torch.tensor([
        ...     [0, 1, 1, 2, 2, 3],
        ...     [1, 0, 2, 1, 3, 2],
        ... ])
        >>> data = Data(edge_index=edge_index, num_nodes=4)
        >>> g = to_networkx(data)
        >>> # A `Data` object is returned
        >>> from_networkx(g)
        Data(edge_index=[2, 6], num_nodes=4)
    """

    G = G.to_directed() if not nx.is_directed(G) else G

    mapping = dict(zip(G.nodes(), range(G.number_of_nodes())))

    edge_index = torch.empty((2, G.number_of_edges()), dtype=torch.long)
    for i, (src, dst) in enumerate(G.edges()):
        edge_index[0, i] = mapping[src]
        edge_index[1, i] = mapping[dst]

    data_dict: Dict[str, Any] = defaultdict(list)
    data_dict["edge_index"] = edge_index

    node_attrs: List[str] = []
    if G.number_of_nodes() > 0:
        node_attrs = list(next(iter(G.nodes(data=True)))[-1].keys())

    edge_attrs: List[str] = []
    if G.number_of_edges() > 0:
        edge_attrs = list(next(iter(G.edges(data=True)))[-1].keys())

    if group_node_attrs is not None and not isinstance(group_node_attrs, list):
        group_node_attrs = node_attrs

    if group_edge_attrs is not None and not isinstance(group_edge_attrs, list):
        group_edge_attrs = edge_attrs

    for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
        if set(feat_dict.keys()) != set(node_attrs):
            raise ValueError("Not all nodes contain the same attributes")
        for key, value in feat_dict.items():
            data_dict[str(key)].append(value)

    for i, (_, _, feat_dict) in enumerate(G.edges(data=True)):
        if set(feat_dict.keys()) != set(edge_attrs):
            raise ValueError("Not all edges contain the same attributes")
        for key, value in feat_dict.items():
            key = f"edge_{key}" if key in node_attrs else key
            data_dict[str(key)].append(value)

    for key, value in G.graph.items():
        if key == "node_default" or key == "edge_default":
            continue  # Do not load default attributes.
        key = f"graph_{key}" if key in node_attrs else key
        data_dict[str(key)] = value

    for key, value in data_dict.items():
        if isinstance(value, (tuple, list)) and isinstance(value[0], torch.Tensor):
            data_dict[key] = torch.stack(value, dim=0)
        else:
            try:
                data_dict[key] = torch.as_tensor(value)
            except Exception:
                pass

    data = Data.from_dict(data_dict)

    if group_node_attrs is not None:
        xs = []
        for key in group_node_attrs:
            x = data[key]
            x = x.view(-1, 1) if x.dim() <= 1 else x
            xs.append(x)
            del data[key]
        data.x = torch.cat(xs, dim=-1)

    if group_edge_attrs is not None:
        xs = []
        for key in group_edge_attrs:
            key = f"edge_{key}" if key in node_attrs else key
            x = data[key]
            x = x.view(-1, 1) if x.dim() <= 1 else x
            xs.append(x)
            del data[key]
        data.edge_attr = torch.cat(xs, dim=-1)

    if data.x is None and data.pos is None:
        data.num_nodes = G.number_of_nodes()

    return data


def prepare_graph(G):

    G = G.copy()
    G = nx.Graph(G)

    for node, data in G.nodes(data=True):
        if "self" in data:
            del data["self"]
    return G
