import json
from logging import Logger
from pathlib import Path
from typing import Any, Literal, Optional, Type, TypeVar, Dict, List, Tuple
from termcolor import colored
import numpy as np
from pydantic import BaseModel
from text_generation import Client
from text_generation.types import Grammar, GrammarType
from primaite import getLogger
from primaite.action import NodeAction
from primaite.agents.agent_abc import AgentSessionABC
from primaite.agents.llm.utils import network_connectivity_desc, obs_diff, obs_view_full, get_obs_act_history_str
from primaite.common.enums import AgentFramework, AgentIdentifier
from primaite.environment.env_state import EnvironmentState
from primaite.environment.primaite_env import Primaite
from primaite.exceptions import LLMGrammarError
from outlines.fsm.json_schema import build_regex_from_schema
from primaite.agents.llm.prompting import (
    SYSTEM_MSG,
    REASON_ACTION_SPACE_NODE_SELECT,
    NODE_ACTION_SELECTION,
    ACTION_INFO,
    AgentReasoningNodeSelection,
    AgentNodeAction,
)

_LOGGER: Logger = getLogger(__name__)
T = TypeVar("T", bound=BaseModel)
MAX_PROMPT_OBS_HISTORY = 20


def format_llama_prompt(system: str, messages: List[Tuple[Literal["user", "assistant"], str]]) -> str:
    prompt = "<|begin_of_text|>"

    # system prompt
    prompt += "<|start_header_id|>system<|end_header_id|>\n\n"
    prompt += system
    prompt += "<|eot_id|>"

    # prev messages
    for role, msg in messages:
        prompt += f"<|start_header_id|>{role}<|end_header_id|>\n\n"
        prompt += msg
        prompt += "<|eot_id|>"

    # prompt llm to answer
    prompt += "<|start_header_id|>assistant<|end_header_id|>"

    return prompt


class LLM:
    def __init__(self, base_url: str, timeout: int = 60) -> None:
        self.client = Client(base_url=base_url, timeout=timeout)

    def generate(self, prompt: str, repetition_penalty: Optional[float] = None, max_new_tokens: int = 1024) -> str:
        response = self.client.generate(
            prompt=prompt, max_new_tokens=max_new_tokens, repetition_penalty=repetition_penalty
        )
        return response.generated_text

    def generate_model(
        self,
        prompt: str,
        model: Type[T],
        repetition_penalty: Optional[float] = None,
        max_new_tokens: int = 1024,
        new_literals: Optional[Dict[str, List]] = None,
    ) -> T:

        model_schema = model.model_json_schema()
        if new_literals:
            # For each set of literals (referred to as enums) passed, add to the schema for the respective property.
            for prop, enum in new_literals.items():
                model_schema["properties"][prop]["enum"] = enum

        # Convert json schema to regex to preserve property order during generation
        model_json = json.dumps(model_schema, indent=2)
        model_regex = build_regex_from_schema(model_json)

        # Generate response for regex
        response = self.client.generate(
            prompt=prompt,
            grammar=Grammar(type=GrammarType.Regex, value=model_regex),
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
        ).generated_text

        try:
            # Parse response
            response_model = model(**json.loads(response))
            _LOGGER.info(f"{colored('Action', 'green')}: {response_model}")
        except json.JSONDecodeError as e:
            # Grammar didn't work
            raise LLMGrammarError(f"LLM did not produce valid grammar JSON schema: {e}. Generated: {response}")
        except Exception as e:
            raise Exception(e)

        return response_model

    def _build_reasoning_prompt(self, env_state: EnvironmentState, env_history: list[EnvironmentState]) -> str:
        prompt = ""

        # Get the action space description
        env = env_state.env
        initial_state = env_history[0]

        node_names = "'" + ", ".join([n.name for n in env.active_nodes]) + "'"  # Nice comma separated list
        service_names = "'" + ", ".join(env.services_list) + "'"

        obs_act_history = get_obs_act_history_str(
            env_history=env_history, env=env_state.env, max_history=MAX_PROMPT_OBS_HISTORY
        )

        # Current observation space changes
        _LOGGER.info(f"{colored('Observed changes', 'yellow')}: {obs_diff(env_state)}\n\n")

        prompt = REASON_ACTION_SPACE_NODE_SELECT.format(
            node_names=node_names,
            service_names=service_names,
            network_connectivity_desc=network_connectivity_desc(initial_state),
            initial_obs_view_full=obs_view_full(initial_state),
            obs_act_history=obs_act_history,
            current_obs_view_full=obs_view_full(env_state),
            current_obs_diff=obs_diff(env_state),
            action_info=ACTION_INFO.format(service_names=service_names),
        )
        messages: List[Tuple[Literal["user", "assistant"], str]] = [("user", prompt)]
        prompt = format_llama_prompt(system=SYSTEM_MSG, messages=messages)

        return prompt

    def _build_action_prompt(
        self, env_state: EnvironmentState, env_history: list[EnvironmentState], node_name: str, reasoning: str
    ) -> str:
        prompt = ""

        # Get the action space description
        env = env_state.env
        initial_state = env_history[0]
        service_names = "'" + ", ".join(env.services_list) + "'"

        obs_act_history = get_obs_act_history_str(
            env_history=env_history, env=env_state.env, max_history=MAX_PROMPT_OBS_HISTORY
        )

        prompt = NODE_ACTION_SELECTION.format(
            node_name=node_name,
            reasoning=reasoning,
            network_connectivity_desc=network_connectivity_desc(initial_state),
            initial_obs_view_full=obs_view_full(initial_state),
            obs_act_history=obs_act_history,
            current_obs_view_full=obs_view_full(env_state),
            current_obs_diff=obs_diff(env_state),
            action_info=ACTION_INFO.format(service_names=service_names),
        )
        messages = [("user", prompt)]
        messages: List[Tuple[Literal["user", "assistant"], str]] = [("user", prompt)]
        prompt = format_llama_prompt(SYSTEM_MSG, messages)

        return prompt

    def predict(self, env_state: EnvironmentState, env_history: list[EnvironmentState]) -> Tuple[int, str, str]:
        env = env_state.env

        # Think and decide which node to act on
        prompt = self._build_reasoning_prompt(env_state=env_state, env_history=env_history)
        agent_reason_select = self.generate_model(
            prompt=prompt,
            model=AgentReasoningNodeSelection,
            repetition_penalty=1.1,
            new_literals={"node_name": [n.name for n in env.active_nodes] + ["NONE"]},
        )
        reasoning = agent_reason_select.reasoning
        node_selection = agent_reason_select.node_name

        # LLM chose to take an action on a node
        if node_selection != "NONE":
            # BUILD PROMPT HERE
            prompt = self._build_action_prompt(
                env_state=env_state, env_history=env_history, reasoning=reasoning, node_name=node_selection
            )
            agent_action = self.generate_model(
                prompt=prompt,
                model=AgentNodeAction,
                repetition_penalty=1.1,
                new_literals={"node_name": [n.name for n in env.active_nodes] + ["NONE"]},
            )
            try:
                action = agent_action.to_node_action(env=env)
            except BaseException:
                _LOGGER.info(f"Invalid LLM action: {agent_action}")
                action = NodeAction(env=env)

        # When the LLM chose to take no action
        else:
            action = NodeAction(env=env)

        action_id = action.action_id
        return action_id, prompt, reasoning


class LLMAgent(AgentSessionABC):
    def __init__(self, training_config_path, lay_down_config_path):
        super().__init__(training_config_path, lay_down_config_path)
        assert self._training_config.agent_framework == AgentFramework.CUSTOM
        assert self._training_config.agent_identifier == AgentIdentifier.LLM
        self._setup()

    def _setup(self):
        super()._setup()

        if not isinstance(self.session_path, Path):
            self.session_path = Path(self.session_path)

        self._env = Primaite(
            training_config_path=self._training_config_path,
            lay_down_config_path=self._lay_down_config_path,
            session_path=self.session_path,
            timestamp_str=self.timestamp_str,
        )
        self._agent = LLM(base_url="http://192.168.0.8:58084", timeout=120)

        # Keep track of env history
        self.env_history = [EnvironmentState(self._env)]

    def _save_checkpoint(self) -> None:
        _LOGGER.warning("Deterministic agents cannot learn")

    def learn(self):
        _LOGGER.warning("Deterministic agents cannot learn")

    def _calculate_action(self, obs: np.ndarray):
        action, prompt, reasoning = self._calculate_action_info(obs)

        return action

    def _calculate_action_info(self, obs: np.ndarray) -> tuple[int, str | None, str | None]:
        prev_env_state = self.env_history[-1]
        env_state = EnvironmentState(self._env, prev_env_state=prev_env_state)

        action, prompt, reasoning = self._agent.predict(env_state, self.env_history)
        env_state.action_id = action
        self.env_history.append(env_state)

        return action, prompt, reasoning

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

        _LOGGER.info(f"Num services: {self._env.num_services}")

        for _ in range(episodes):
            obs = self._env.reset()
            done, steps, rew = False, 0, 0
            while steps < time_steps and not done:

                action = self._calculate_action(obs)

                obs, rewards, done, info = self._env.step(action=action)
                steps += 1
                # rew += rewards

        self._env._write_av_reward_per_episode()  # noqa
        self._env.close()
        super().evaluate()

    def _get_latest_checkpoint(self):
        pass

    @classmethod
    def load(cls, path):
        pass

    def save(self):
        return None

    def export(self) -> None:
        return None
