from functools import cached_property
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from primaite.common.enums import NodeFileSystemAction, NodeHardwareAction, NodePropertyAction, NodeSoftwareAction
from primaite.common.service import Service
from primaite.environment import Primaite
from primaite.nodes import Node, ServiceNode


# TODO - make sure action can actually be applied, given the current env_state
class NodeAction(BaseModel):
    env: Primaite
    node: Node | None = None
    node_property: NodePropertyAction = NodePropertyAction.NONE
    property_action: NodeHardwareAction | NodeSoftwareAction | NodeFileSystemAction | None = None
    service: Service | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_numerical(cls, env: Primaite, node_id: int, property_id: int, action_id: int, service_id: int):

        if node_id == 0:
            return NodeAction(env=env)

        else:
            try:
                node = env.nodes[str(node_id)]
            except:
                raise ValueError(f"Provided node_id {node_id} could not be found in the environment specs.")

        try:
            node_property = NodePropertyAction(property_id)
        except:
            raise ValueError(f"Provided property_id {property_id} is not valid. See the NodePropertyAction enum.")

        match node_property:
            case NodePropertyAction.NONE:
                action_enum = None
            case NodePropertyAction.HARDWARE:
                action_enum = NodeHardwareAction
            case NodePropertyAction.SOFTWARE:
                action_enum = NodeSoftwareAction
            case NodePropertyAction.SERVICE:
                action_enum = NodeSoftwareAction
            case NodePropertyAction.FILE_SYSTEM:
                action_enum = NodeFileSystemAction
        try:
            if action_enum is None:
                property_action = None
            else:
                property_action = action_enum(action_id)
        except:
            raise ValueError(f"Provided action_id {action_id} is not valid. See the {action_enum} enum.")

        if node_property == NodePropertyAction.SERVICE:

            # NOTE: the documentation is conflicting here... they claim that a service id of 0 represents nothing but if you look at how the code works the service id 0 just corresponds to the first service... so commenting this one out

            # if service_id == 0:
            #     raise ValueError("Need to select a service when taking an action on the SERVICE node property.")

            if not isinstance(node, ServiceNode):
                raise ValueError(
                    f"Cannot take an action on the SERVICE node property for the non-ServiceNode {node.name}."
                )

            try:
                service_name = env.services_list[service_id]
            except:
                raise ValueError("Provided service_id {service_id} could not be found in the environment specs.")
            try:
                service = node.services[service_name]
            except:
                raise ValueError(f"The node {node.name} does not own the service {service_name}")
        else:
            service = None

        return cls(
            env=env,
            node=node,
            node_property=node_property,
            property_action=property_action,
            service=service,
        )

    @classmethod
    def from_text(cls, env: Primaite, node_name: str, node_property: str, property_action: str, service_name: str):

        # We don't do any validation here since it's all done in the from_numerical class method
        # We artificially induce invalid ids if we cannot resolve them from the textual representations given
        if node_name == "NONE":
            node_id = 0
        else:
            node_id = next(
                (int(n.node_id) for k, n in env.nodes.items() if n.name == node_name), len(env.nodes.items()) + 10
            )

        try:
            property_id = NodePropertyAction[node_property].value
            match NodePropertyAction(property_id):
                case NodePropertyAction.NONE:
                    action_enum = None
                case NodePropertyAction.HARDWARE:
                    action_enum = NodeHardwareAction
                case NodePropertyAction.SOFTWARE:
                    action_enum = NodeSoftwareAction
                case NodePropertyAction.SERVICE:
                    action_enum = NodeSoftwareAction
                case NodePropertyAction.FILE_SYSTEM:
                    action_enum = NodeFileSystemAction
            if action_enum is not None:
                action_id = action_enum[property_action].value
            else:
                action_id = 0
        except:
            property_id = -1
            action_id = -1

        try:
            service_id = env.services_list.index(service_name)
        except:
            service_id = len(env.services_list) + 10

        return NodeAction.from_numerical(
            env=env, node_id=node_id, property_id=property_id, action_id=action_id, service_id=service_id
        )

    @classmethod
    def from_id(cls, env: Primaite, action_id: int):
        numerical_repr = env.action_dict[action_id]
        return NodeAction.from_numerical(env, *numerical_repr)

    @property
    def numerical(self) -> list[int]:
        node_id = 0 if self.node is None else int(self.node.node_id)

        property_id = self.node_property.value

        action_id = 0 if self.property_action is None else self.property_action.value

        service_id = 0 if self.service is None else self.env.services_list.index(self.service.name)

        return [node_id, property_id, action_id, service_id]

    @property
    def as_text(self) -> list[str]:
        node_name = "NONE" if self.node is None else self.node.name
        node_property = self.node_property.name
        property_action = "NONE" if self.property_action is None else self.property_action.name
        service_name = "NONE" if self.service is None else self.service.name
        return [node_name, node_property, property_action, service_name]

    @cached_property
    def action_id(self) -> int:
        action_dict = self.env.action_dict
        try:
            id = next(k for k, v in action_dict.items() if v == self.numerical)
        except BaseException:
            print("numerical repr of action:")
            print(self.numerical)
            print("action dict:")
            for k, v in action_dict.items():
                print(v)
            raise ValueError(f"Could not find a valid id in the action dict for action {self.as_text}.")

        return id

    def has_effect(self) -> bool:
        return (
            self.node is not None and self.node_property != NodePropertyAction.NONE and self.property_action is not None
        )

    def verbose(self, colored=False) -> str:
        if not self.has_effect():
            return "No action."
        assert self.property_action is not None
        act = self.property_action.name
        if colored:
            act = f":red[{act}]"

        assert self.node is not None
        node_name = self.node.name
        if colored:
            node_name = f":blue[{node_name}]"

        state = self.node_property.name

        if self.node_property == NodePropertyAction.SERVICE:
            assert self.service is not None
            state = self.service.name + " " + state

        if colored:
            state = f":violet[{state}]"

        return f"Action {act} was taken on node {node_name}'s {state} State"
