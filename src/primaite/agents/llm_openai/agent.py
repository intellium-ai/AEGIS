from __future__ import annotations

import json
from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, TypeVar
from dotenv import load_dotenv

import numpy as np
from outlines.fsm.json_schema import build_regex_from_schema
from pydantic import BaseModel
from termcolor import colored
from text_generation import Client
from text_generation.types import Grammar, GrammarType

from primaite import getLogger
from primaite.action import NodeAction
from primaite.agents.aegis.modules.openai import OpenAIClient
from primaite.agents.agent_abc import AgentSessionABC
from primaite.agents.llm.observation import get_obs_act_history_str, network_connectivity_desc, ObservedState
from primaite.agents.llm.prompting import (
    ACTION_INFO,
    AgentNodeAction,
    AgentReasoningNodeSelection,
    NODE_ACTION_SELECTION,
    REASON_ACTION_SPACE_NODE_SELECT,
    SYSTEM_MSG,
)
from primaite.common.enums import AgentFramework, AgentIdentifier
from primaite.environment.primaite_env import Primaite
from primaite.exceptions import LLMGrammarError

_LOGGER: Logger = getLogger(__name__)
T = TypeVar("T", bound=BaseModel)
MAX_PROMPT_OBS_HISTORY = 20


class LLM:
    def __init__(self, api_key) -> None:
        self.client = OpenAIClient(api_key=api_key)

    def generate(self, prompt: str) -> str:
        response = self.client.generate(prompt=prompt)
        return response
    
    def generate_model(
        self, prompt: str, grammar: Type[T], max_new_tokens: int = 2048, model: str = "gpt-3.5-turbo"
    ) -> Type[T]:

        messages = [{"role": "user", "content": prompt}]
        model_schema = self._create_openai_schema(grammar)
        function_call = {"name": grammar.__name__}
        functions = [model_schema]

        # Get response from the api
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore
            stream=False,
            functions=functions,
            function_call=function_call,
            max_tokens=max_new_tokens,
        )
        return self._construct_model(grammar, model_schema, response)

    def _build_reasoning_prompt(self, obs_state: ObservedState, obs_history: list[ObservedState]) -> str:
        prompt = ""

        # Get the action space description
        initial_state = obs_history[0]
        network = initial_state.network

        node_names = "'" + ", ".join([n.name for n in network.active_nodes]) + "'"  # Nice comma separated list
        service_names = "'" + ", ".join(network.service_names) + "'"

        obs_act_history = get_obs_act_history_str(obs_history=obs_history, max_history=MAX_PROMPT_OBS_HISTORY)

        # Current observation space changes
        _LOGGER.info(f"{colored('Observed changes', 'yellow')}: {obs_state.changes_str}\n\n")

        prompt = REASON_ACTION_SPACE_NODE_SELECT.format(
            node_names=node_names,
            service_names=service_names,
            network_connectivity_desc=network_connectivity_desc(network),
            initial_obs_view_full=initial_state.format(),
            obs_act_history=obs_act_history,
            current_obs_view_full=obs_state.format(),
            current_obs_diff=obs_state.changes_str,
            action_info=ACTION_INFO.format(service_names=service_names),
        )
        messages: List[Tuple[Literal["user", "assistant"], str]] = [("user", prompt)]
        prompt = format_llama_prompt(system=SYSTEM_MSG, messages=messages)

        return prompt

    def _build_action_prompt(
        self, obs_state: ObservedState, obs_history: list[ObservedState], node_name: str, reasoning: str
    ) -> str:
        prompt = ""

        # Get the action space description
        initial_state = obs_history[0]
        network = initial_state.network
        service_names = "'" + ", ".join(network.service_names) + "'"

        obs_act_history = get_obs_act_history_str(obs_history=obs_history, max_history=MAX_PROMPT_OBS_HISTORY)

        prompt = NODE_ACTION_SELECTION.format(
            node_name=node_name,
            reasoning=reasoning,
            network_connectivity_desc=network_connectivity_desc(network),
            initial_obs_view_full=initial_state.format(),
            obs_act_history=obs_act_history,
            current_obs_view_full=obs_state.format(),
            current_obs_diff=obs_state.changes_str,
            action_info=ACTION_INFO.format(service_names=service_names),
        )
        messages = [("user", prompt)]
        messages: List[Tuple[Literal["user", "assistant"], str]] = [("user", prompt)]
        prompt = format_llama_prompt(SYSTEM_MSG, messages)

        return prompt

    def _construct_model(
        self,
        model: Type[T],
        schema: Dict[str, Any],
        completion: Any,
        throw_error=True,
    ):
        """Execute the function from the response of an openai chat completion

        Parameters:
            model (Type[T]): The model to construct using the response.
            schema (Dict[str, Any]): The openai compatible schema to use.
            completion (Any): The response from an openai chat completion.
            throw_error (bool): Whether to throw an error if the function call is not detected. Defaults to True.

        Returns:
            model (OpenAISchema): An instance of the class
        """
        message = completion.choices[0].message

        if throw_error:
            if not hasattr(message, "function_call") or message.function_call.name != schema["name"]:
                raise Exception("No function call detected")

        function_call = message.function_call
        try:
            arguments = json.loads(function_call.arguments, strict=False)
        except json.JSONDecodeError:
            raise Exception("Unable to parse LLM output as the LLM likely generated incomplete JSON")
        return model(**arguments)

    def remove_key_from_dict(self, d: dict[str, Any], remove_key: str) -> None:
        """
        Remove a key from a dictionary recursively

        Args:
            d (Dict[str, Any]): The dictionary
            remove_key (str): The key to remove
        """

        if isinstance(d, dict):
            for key in list(d.keys()):
                if key == remove_key:
                    del d[key]
                else:
                    self.remove_key_from_dict(d[key], remove_key)

    # These are some util functions I stole from a library called openai_function_call that does the json related grammar stuff.
    def _create_openai_schema(self, model: Type[T]) -> Dict[str, Any]:
        """
        Return the schema in the format of OpenAI's schema as jsonschema

        Returns:
            model_json_schema (dict): A dictionary in the format of OpenAI's schema as jsonschema
        """

        schema = model.model_json_schema()
        parameters = {k: v for k, v in schema.items() if k not in ("title", "description")}
        parameters["required"] = sorted(k for k, v in parameters["properties"].items() if "default" not in v)

        if "description" not in schema:
            schema["description"] = (
                f"Correctly extracted `{model.__name__}` with all the required parameters with correct types"  # type: ignore
            )

        self.remove_key_from_dict(parameters, "additionalProperties")
        self.remove_key_from_dict(parameters, "title")
        return {
            "name": schema["title"],
            "description": schema["description"],
            "parameters": parameters,
        }

    def predict(self, obs_state: ObservedState, obs_history: list[ObservedState]) -> Tuple[int, str, str]:

        # Think and decide which node to act on
        prompt = self._build_reasoning_prompt(obs_state=obs_state, obs_history=obs_history)
        network = obs_state.network
        agent_reason_select = self.generate_model(
            prompt=prompt,
            model=AgentReasoningNodeSelection,
            repetition_penalty=1.1,
            new_literals={"node_name": [n.name for n in network.active_nodes] + ["NONE"]},
        )
        reasoning = agent_reason_select.reasoning
        node_selection = agent_reason_select.node_name

        # LLM chose to take an action on a node
        if node_selection != "NONE":
            # BUILD PROMPT HERE
            prompt = self._build_action_prompt(
                obs_state=obs_state, obs_history=obs_history, reasoning=reasoning, node_name=node_selection
            )
            agent_action = self.generate_model(
                prompt=prompt,
                model=AgentNodeAction,
                repetition_penalty=1.1,
                new_literals={"node_name": [n.name for n in network.active_nodes] + ["NONE"]},
            )
            try:
                action = agent_action.to_node_action(network=network)
            except BaseException:
                _LOGGER.info(f"Invalid LLM action: {agent_action}")
                action = NodeAction(network=network)

        # When the LLM chose to take no action
        else:
            action = NodeAction(network=network)

        action_id = action.action_id
        return action_id, prompt, reasoning