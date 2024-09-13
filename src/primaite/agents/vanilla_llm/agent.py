from __future__ import annotations

import json
from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, TypeVar

import numpy as np
from outlines.fsm.json_schema import build_regex_from_schema
from pydantic import BaseModel
from termcolor import colored
from text_generation import Client
from text_generation.types import Grammar, GrammarType

from primaite import getLogger
from primaite.action import NodeAction
from transformers import AutoTokenizer
from fireworks.client import Fireworks
from primaite.agents.agent_abc import AgentSessionABC
from primaite.agents.vanilla_llm.observation import get_obs_act_history_str, network_connectivity_desc, ObservedState
from primaite.agents.vanilla_llm.prompting import (
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


class FireworksLLM:
    def __init__(self, api_key, model="accounts/fireworks/models/llama-v3p1-8b-instruct") -> None:
        self.client = Fireworks(api_key=api_key)
        self.model = model

    def generate(self, system, prompt, max_new_tokens=1024, repetition_penalty: float = 1.1):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = self.client.chat.completions.create(
            model=self.model,
            prompt_or_messages=messages,
            max_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
        )
        return response.choices[0].message.content

    def generate_model(
        self,
        system,
        prompt,
        grammar,
        max_new_tokens=1024,
        new_literals: Optional[Dict[str, List]] = None,
        repetition_penalty: float = 1.1,
    ):
        model_schema = grammar.model_json_schema()
        if new_literals:
            # For each set of literals (referred to as enums) passed, add to the schema for the respective property.
            for prop, enum in new_literals.items():
                model_schema["properties"][prop]["enum"] = enum

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = self.client.chat.completions.create(
            model=self.model,
            prompt_or_messages=messages,
            max_tokens=max_new_tokens,
            response_format={"type": "json_object", "schema": model_schema},
            repetition_penalty=repetition_penalty,
        )
        try:
            response = json.loads(response.choices[0].message.content)
        except:
            response = {}
        return grammar(**response)

class HFClient:
    def __init__(self, base_url: str, timeout: int =120) -> None:
        self.client = Client(base_url=base_url, timeout=timeout)
        self.tokenizer = AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM-1.7B-Instruct')
        
    def generate_model(self, system: str, prompt: str, grammar, max_new_tokens: int = 1024, new_literals: Optional[Dict[str, List]] = None, repetition_penalty: float = 1.1) -> Type[T]:
        
        model_json = grammar.model_json_schema()
        if new_literals:
            # For each set of literals (referred to as enums) passed, add to the schema for the respective property.
            for prop, enum in new_literals.items():
                model_json["properties"][prop]["enum"] = enum
        gram = Grammar(type=GrammarType.Json, value=model_json)
        messages = []
        prompt = prompt[:2000]
        messages.append({'role': 'system', 'content': system})
        messages.append({'role': 'user', 'content': prompt})
        prompt_str = self.tokenizer.apply_chat_template(messages, tokenize=False)
        resp = self.client.generate(prompt_str, grammar=gram, max_new_tokens=max_new_tokens, repetition_penalty=repetition_penalty).generated_text
        
        try:
            response_model = grammar(**json.loads(resp))
        except:
            response_model = grammar()
        print('WE did it with SMOLLM!!')
        return response_model
        
        
        
        
        
def _build_reasoning_prompt(obs_state: ObservedState, obs_history: list[ObservedState]) -> str:
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

    return prompt

def _build_action_prompt(
    obs_state: ObservedState, obs_history: list[ObservedState], node_name: str, reasoning: str
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

    return prompt
    
def predict(llm: Tuple[FireworksLLM, HFClient], obs_state: ObservedState, obs_history: list[ObservedState]) -> Tuple[int, str, str]:

    # Think and decide which node to act on
    prompt = _build_reasoning_prompt(obs_state=obs_state, obs_history=obs_history)
    network = obs_state.network
    agent_reason_select = llm.generate_model(
        prompt=prompt,
        system=SYSTEM_MSG,
        grammar=AgentReasoningNodeSelection,
        new_literals={"node_name": [n.name for n in network.active_nodes] + ["NONE"]},
        repetition_penalty=1.1,
    )
    
    # Handle grammar errors
    if not agent_reason_select.reasoning:
        reasoning = 'NONE'
    else:
        reasoning = agent_reason_select.reasoning
        
    if not agent_reason_select.node_name:
        node_selection = 'NONE'
    else:
        node_selection = agent_reason_select.node_name

    # LLM chose to take an action on a node
    if node_selection != "NONE":
        # BUILD PROMPT HERE
        prompt = _build_action_prompt(
            obs_state=obs_state, obs_history=obs_history, reasoning=reasoning, node_name=node_selection
        )
        # rep penalty 1.1
        agent_action = llm.generate_model(
            prompt=prompt,
            system=SYSTEM_MSG,
            grammar=AgentNodeAction,
            new_literals={"node_name": [n.name for n in network.active_nodes] + ["NONE"]},
            repetition_penalty=1.1,
        )
        try:
            action = agent_action.to_node_action(network=network)
            action_id = action.action_id
        except BaseException:
            _LOGGER.info(f"Invalid LLM action: {agent_action}")
            action = NodeAction(network=network)
            action_id = 0

    # When the LLM chose to take no action
    else:
        action = NodeAction(network=network)
        action_id = 0
    return action_id, prompt, reasoning


class LLMAgent(AgentSessionABC):

    @dataclass
    class ActionInfo(AgentSessionABC.ActionInfo):
        prompt: str
        reasoning: str

    def __init__(self, training_config_path, lay_down_config_path, fireworks_api_key: str):
        super().__init__(training_config_path, lay_down_config_path)
        assert self._training_config.agent_framework == AgentFramework.CUSTOM
        assert self._training_config.agent_identifier == AgentIdentifier.LLM
        self._setup(fireworks_api_key)

    def _setup(self, fireworks_api_key: str):
        super()._setup()

        if not isinstance(self.session_path, Path):
            self.session_path = Path(self.session_path)

        self._env = Primaite(
            training_config_path=self._training_config_path,
            lay_down_config_path=self._lay_down_config_path,
            session_path=self.session_path,
            timestamp_str=self.timestamp_str,
        )
        
        self._agent = HFClient(base_url='http://192.168.0.68:58084', timeout=120)
        # self._agent = FireworksLLM(api_key=fireworks_api_key)

        # Keep track of env history
        self.obs_history = [ObservedState.from_env(self._env)]

    def _save_checkpoint(self) -> None:
        _LOGGER.warning("Deterministic agents cannot learn")

    def learn(self):
        _LOGGER.warning("Deterministic agents cannot learn")

    def _calculate_action(self, obs: np.ndarray):
        action, _ = self.calculate_action_info(obs)

        return action

    def calculate_action_info(self, obs: np.ndarray) -> tuple[int, ActionInfo]:
        prev_obs_state = self.obs_history[-1]
        curr_obs_state = ObservedState.from_env(self._env, prev_obs_state)

        action_id, prompt, reasoning = predict(llm=self._agent, obs_state=curr_obs_state, obs_history=self.obs_history)
        curr_obs_state.action = NodeAction.from_id(network=curr_obs_state.network, action_id=action_id)
        self.obs_history.append(curr_obs_state)

        info = LLMAgent.ActionInfo(prompt=prompt, reasoning=reasoning)

        return action_id, info

    def evaluate(
        self,
        **kwargs: Any,
    ) -> None:
        """
        Evaluate the agent.

        :param kwargs: Any agent-specific key-word args to be passed.
        """
        time_steps = self._training_config.num_eval_steps
        episodes = self._training_config.num_eval_episodes
        self._env.set_as_eval()
        self.is_eval = True
        ep_rewards = []
        _LOGGER.info(f"Num services: {self._env.num_services}")

        for _ in range(episodes):
            obs = self._env.reset()
            done, steps, rew = False, 0, 0
            while steps < time_steps and not done:

                action = self._calculate_action(obs)

                obs, rewards, done, info = self._env.step(action=action)
                steps += 1
                # rew += rewards
            ep_rewards.append(self._env.average_reward)
        self._env._write_av_reward_per_episode()  # noqa
        self._env.close()
        super().evaluate()
        return np.mean(ep_rewards)
    def _get_latest_checkpoint(self):
        pass

    @classmethod
    def load(cls, path):
        pass

    def save(self):
        return None

    def export(self) -> None:
        return None
