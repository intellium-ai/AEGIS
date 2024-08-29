# © Crown-owned copyright 2023, Defence Science and Technology Laboratory UK
from collections import defaultdict
from typing import Any, Dict, List, Literal, Optional, Union

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


def create_node_action_dict(num_nodes: int, num_services: int) -> Dict[int, List[int]]:
    """
    Creates a dictionary mapping each possible discrete action to more readable multidiscrete action.

    Note: Only actions that have the potential to change the state exist in the mapping (except for key 0)

    example return:
    {0: [1, 0, 0, 0],
    1: [1, 1, 1, 0],
    2: [1, 1, 2, 0],
    3: [1, 1, 3, 0],
    4: [1, 2, 1, 0],
    5: [1, 3, 1, 0],
    ...
    }
    """
    # Terms (for node action space):
    # [0, num nodes] - node ID (0 = nothing, node ID)
    # [0, 4] - what property it's acting on (0 = nothing, state, SoftwareState, service state, file system state) # noqa
    # [0, 3] - action on property (0 = nothing, On / Scan, Off / Repair, Reset / Patch / Restore) # noqa
    # [0, num services] - resolves to service ID (0 = nothing, resolves to service) # noqa
    # reserve 0 action to be a nothing action
    actions = {0: [0, 0, 0, 0]}
    action_key = 1
    for node in range(1, num_nodes + 1):
        # 4 node properties (NONE, OPERATING, OS, SERVICE, FILE_SYSTEM)
        for node_property in range(5):
            # Node Actions either:
            # (NONE, ON, OFF, RESET) - operating state OR (NONE, PATCH) - OS/service state
            # Use MAX to ensure we get them all
            for node_action in range(4):
                for service_state in range(num_services):
                    action = [node, node_property, node_action, service_state]
                    # check to see if it's a nothing action (has no effect)
                    if is_valid_node_action(action):
                        actions[action_key] = action
                        action_key += 1

    return actions


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


def is_valid_node_action(action: List[int]) -> bool:
    """
    Is the node action an actual valid action.

    Only uses information about the action to determine if the action has an effect

    Does NOT consider:
    - Node ID not valid to perform an operation - e.g. selected node has no service so cannot patch
    - Node already being in that state (turning an ON node ON)

    :param action: Agent action, formatted as a list of ints, for more information check out
        `primaite.environment.primaite_env.Primaite`
    :type action: List[int]
    :return: Whether the action is valid
    :rtype: bool
    """
    action_r = transform_action_node_readable(action)

    node_property = action_r[1]
    node_action = action_r[2]

    # print("node property", node_property, "\nnode action", node_action)

    if node_property == "NONE":
        return False
    if node_action == "NONE":
        return False
    if node_property == "OPERATING" and node_action == "PATCH":
        # Operating State cannot PATCH
        return False
    if node_property != "OPERATING" and node_action not in [
        "NONE",
        "PATCH",
    ]:
        # Software States can only do Nothing or Patch
        return False
    return True


def is_valid_acl_action(action: List[int], implicit_permission: RulePermissionType) -> bool:
    """
    Is the ACL action an actual valid action.

    Only uses information about the action to determine if the action has an effect.

    Does NOT consider:
        - Trying to create identical rules
        - Trying to create a rule which is a subset of another rule (caused by "ANY")

    :param action: Agent action, formatted as a list of ints, for more information check out
        `primaite.environment.primaite_env.Primaite`
    :type action: List[int]
    :return: Whether the action is valid
    :rtype: bool
    """
    action_r = transform_action_acl_readable(action)

    action_decision = action_r[0]
    action_permission = action_r[1]
    action_source_id = action_r[2]
    action_destination_id = action_r[3]

    if action_decision == "NONE":
        return False
    if action_source_id == action_destination_id and action_source_id != "ANY" and action_destination_id != "ANY":
        # ACL rule towards itself
        return False
    if action_permission == "DENY" and implicit_permission == RulePermissionType.DENY:
        # DENY is unnecessary, we can create and delete allow rules instead
        # No allow rule = blocked/DENY by feault. ALLOW overrides existing DENY.
        return False
    elif action_permission == "ALLOW" and implicit_permission == RulePermissionType.ALLOW:
        # Same reason as above.
        return False

    return True


def is_valid_acl_action_extra(action: List[int], implicit_permission: RulePermissionType) -> bool:
    """
    Harsher version of valid acl actions, does not allow action.

    :param action: Agent action, formatted as a list of ints, for more information check out
        `primaite.environment.primaite_env.Primaite`
    :type action: List[int]
    :return: Whether the action is valid
    :rtype: bool
    """
    if is_valid_acl_action(action, implicit_permission=implicit_permission) is False:
        return False

    action_r = transform_action_acl_readable(action)
    action_protocol = action_r[4]
    action_port = action_r[5]

    # Don't allow protocols or ports to be ANY
    # in the future we might want to do the opposite, and only have ANY option for ports and service
    if action_protocol == "ANY":
        return False
    if action_port == "ANY":
        return False

    return True
