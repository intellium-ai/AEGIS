from typing import Literal

from pydantic import BaseModel, Field
from termcolor import colored
from primaite.action import NodeAction
from primaite.environment import Primaite


class AgentNodeAction(BaseModel):
    node_name: str = Field("NONE", description="Node to apply action to.")
    node_property: Literal["NONE", "HARDWARE", "SOFTWARE", "FILE_SYSTEM", "SERVICE"] = Field(
        "NONE", description="Node property to apply action to."
    )
    property_action: str = Field("NONE", description="Action to take on the given node property.")
    service_name: str = Field(
        "NONE", description="Service to apply action to. Only applicable if node_property is SERVICE."
    )

    def to_node_action(self, env: Primaite) -> NodeAction:
        return NodeAction.from_text(
            env=env,
            node_name=self.node_name,
            node_property=self.node_property,
            property_action=self.property_action,
            service_name=self.service_name,
        )


class AgentReasoningNodeSelection(BaseModel):
    reasoning: str = Field(..., description="Communication of observational understanding of any actions required.")
    node_name: str = Field(
        "NONE", description="Selection of a node of highest priority for which to perform an action on."
    )


SYSTEM_MSG = "You are a cybersecurity network defence agent. Your mission is to protect a network against offensive attacks. You have a limited view of the network environment, known as an observation space. The network is made up of nodes (e.g. computers, switches) and links between some of the nodes, through which information is transmitted using standard protocols. The current link traffic is displayed as a percentage of its total bandwidth, for each possible protocol. An attacker is trying to compromise the network, by either directly incapacitating the nodes HARDWARE, SOFTWARE or FILE_SYSTEM states, or indirectly overwhelming the nodes through Denial-of-Service type attacks.\n"

REASON_ACTION_SPACE_NODE_SELECT = """
This is the initial configuration of the network:
{network_connectivity_desc}

Initial
{initial_obs_view_full}

This is the history of offensive action observations and defensive actions you have taken at each step. If nothing happened at a specific step, it is omitted from the history:
{obs_act_history}

Here is an overview of the current observation space:
{current_obs_view_full}
Changes that have occured since last observation are:
{current_obs_diff}

As an agent, you are able to influence the state of this node by switching it on or off, resetting it, patching software or patching any of its services.

Please think about the network configuration and the state of each node. Think about which nodes are most vulnerable to attack and which node requires action the most in order to stop the attack and prevent further spread. Provide your reasoning statement and select a node by name to perform a defensive action on. If no action is required because all is well, you can simply say 'NONE', but always provide a reasoning statement.

If action is not required at the moment, set NONE as the node_name, but always reason over the state of the network.
Note that actions are expensive and can negatively impact the environment if used improperly. For instance, a server which is turned off cannot receive requests from the users and will decrease the reward.  

If choosing to take an action, you must select a node by name from the following list: {node_names}.


For your information, the following actions are available for selection later. Always take note of any action constraints outlined in the description provided.

{action_info}

Your output should be in the following format:
{{'reasoning': 'Reason for node selection', 'node_name': 'NODE_NAME'}}
Reasoning and node selection: 
"""

NODE_ACTION_SELECTION = """
This is the initial configuration of the network:
{network_connectivity_desc}

Initial
{initial_obs_view_full}

This is the history of offensive action observations and defensive actions you have taken at each step. If nothing happened at a specific step, it is omitted from the history:
{obs_act_history}

Here is an overview of the current observation space:
{current_obs_view_full}
Changes that have occured since last observation are:
{current_obs_diff}

You have already thought about the information provided above, and have made the decision to take an action on {node_name} due to the following reasoning:
{reasoning}

As an agent, you are able to influence the state of this node by switching it on or off, resetting it, patching software or patching any of its services.

Your task is to use an action from the given list of possible actions.
Always take note of any action constraints outlined in the description provided.

{action_info}

Description: Patches the service for a number of steps, after which the status of the service returns to 'GOOD'.

Please take a suitable action on the given node.
Action: 
"""

ACTION_INFO = """## HARDWARE Actions:
Action: {{'node_name':'NODE_NAME', 'node_property':'HARDWARE', 'property_action':'TURN_ON'}}
Description: If it is currently off, will turn it on.

Action: {{'node_name':'NODE_NAME', 'node_property':'HARDWARE', 'property_action':'TURN_OFF'}}
Description: If it is currently on, will turn it off.

Action: {{'node_name':'NODE_NAME', 'node_property':'HARDWARE', 'property_action':'RESET'}}
Description: Resets the hardware after a number of steps. Only works if the node is turned on. Resets the status of the software, file system and services back to 'GOOD'.

## SOFTWARE Actions:
Action: {{'node_name':'NODE_NAME', 'node_property':'SOFTWARE', 'property_action':'PATCH'}}
Description: Patches the software for a number of steps, after which the status of software returns to 'GOOD'.

## SERVICE Actions:
The following actions are only applicable if the node owns services. If choosing to take a SERVICE action, you must select the SERVICE by name from the following list: {service_names}. Be mindful that a node may own only a subset of these services. Assuming the selected service name is 'SERVICE_NAME', the following are service actions you can take: 

Action: {{'node_name':'NODE_NAME', 'node_property':'SERVICE', 'property_action':'PATCH', 'service_name':'SERVICE_NAME'}}
Description: Patches the service for a number of steps, after which the status of the service returns to 'GOOD'."""
# (JOHN) - File system action descriptions incase we want to use them again in the future.
# ## FILE_SYSTEM Actions:
# Action: {{'node_name':'NODE_NAME', 'node_property':'FILE_SYSTEM', 'property_action':'SCAN'}}
# Description: Scan the file system to reveal the actual status of the file system. After a number of steps, the observed nodes file system status will be updated to reflect the true status.

# Action: {{'node_name':'NODE_NAME', 'node_property':'FILE_SYSTEM', 'property_action':'REPAIR'}}
# Description: Repairs the file system, setting it back to 'GOOD' after a number of steps. Cannot be applied if file system is 'DESTROYED'.


# Action: {{'node_name':'NODE_NAME', 'node_property':'FILE_SYSTEM', 'property_action':'RESTORE'}}
# Description: Restores the file system, setting it back to 'GOOD' after a number of steps. It can be applied even if the file system is 'DESTROYED', but it is more expensive to use than 'REPAIR'.
