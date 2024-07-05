from collections import defaultdict
from logging import Logger
from pathlib import Path
from typing import Any, List, Literal, Tuple, TypeVar

import numpy as np
import torch
import regex
import wandb
from pydantic import BaseModel
from termcolor import colored
from transformers import AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from peft.tuners.lora import LoraConfig
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from termcolor import colored
from primaite import getLogger
from primaite.action import NodeAction
from primaite.agents.agent_abc import AgentSessionABC
from primaite.agents.llm.prompting import (
    ACTION_INFO,
    ACTION_SELECTION,
    REASON_ACTION_SPACE_NODE_SELECT,
    SYSTEM_MSG,
)
from primaite.agents.llm.utils import (
    get_obs_act_history_str,
    network_connectivity_desc,
    obs_diff,
    obs_view_full,
)
from primaite.common.enums import AgentFramework, AgentIdentifier
from primaite.environment.env_state import EnvironmentState
from primaite.environment.primaite_env import Primaite

_LOGGER: Logger = getLogger(__name__)
T = TypeVar("T", bound=BaseModel)
MAX_PROMPT_OBS_HISTORY = 20


def format_llama_prompt(system: str, messages: List[Tuple[Literal["user", "assistant"], str]]) -> str:
    prompt = "<s>"

    prompt += system

    # prev messages
    for _, msg in messages:
        prompt += "\n\n"
        prompt += msg

    # prompt llm to answer
    return prompt


class TrainableLLM:
    def __init__(self, model_name: str, timeout: int = 60) -> None:
        self._init_llm(model_name)

    def _init_llm(self, model_name: str):
        ppo_config = PPOConfig(
            model_name=model_name,
            learning_rate=1e-4,
            batch_size=1,
            mini_batch_size=1,
        )

        peft_config = LoraConfig(task_type="CAUSAL_LM", r=2, lora_dropout=0.1)

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            peft_config=peft_config,
            device_map = "auto"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.ppo_trainer: PPOTrainer = PPOTrainer(
            config=ppo_config,
            model=self.model,
            tokenizer=self.tokenizer,
        )  # type: ignore

    def generate(self, prompt: str) -> str:
        generation_kwargs = {
            "min_length": 5,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
            "max_new_tokens": 8,
            "temperature": 1,
        }

        tokens = torch.tensor(self.tokenizer.encode(prompt)).to("cuda")

        with torch.no_grad():
            gen_tokens = self.ppo_trainer.generate(tokens, **generation_kwargs)

        response = self.tokenizer.decode(gen_tokens[0][len(tokens) :])
        return response

    def _build_action_prompt(
        self,
        env_state: EnvironmentState,
        env_history: list[EnvironmentState],
    ) -> str:
        prompt = ""

        # Get the action space description
        env = env_state.env
        initial_state = env_history[0]
        service_names = "'" + ", ".join(env.services_list) + "'"

        obs_act_history = get_obs_act_history_str(
            env_history=env_history, env=env_state.env, max_history=MAX_PROMPT_OBS_HISTORY
        )

        prompt = ACTION_SELECTION.format(
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

        # Think and decide which node to act on
        prompt = self._build_action_prompt(env_state=env_state, env_history=env_history)

        response = self.generate(prompt=prompt)
        print(response)
        digits = regex.findall(r"\d+", response)
        action = int(digits[-1]) if digits else 0
        return action, prompt, "reason"


class TrainableLLMAgent(AgentSessionABC):
    def __init__(self, training_config_path, lay_down_config_path):
        super().__init__(training_config_path, lay_down_config_path)
        assert self._training_config.agent_framework == AgentFramework.CUSTOM
        assert self._training_config.agent_identifier == AgentIdentifier.TRAINABLE_LLM
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
        self._agent = TrainableLLM(model_name="mistralai/Mistral-7B-v0.1", timeout=120)

        # Keep track of env history
        self.env_history = [EnvironmentState(self._env)]

    def _save_checkpoint(self) -> None:
        _LOGGER.warning(colored("Saving not implemented yet", color="light_red"))

    def learn(
        self,
        **kwargs: Any,
    ) -> None:
        """
        Train the agent.

        :param kwargs: Any agent-specific key-word args to be passed.
        """
        time_steps = self._training_config.num_train_steps
        episodes = self._training_config.num_train_episodes
        self.is_eval = False

        _LOGGER.info(f"Beginning learning for {episodes} episodes @" f" {time_steps} time steps...")

        for ep in range(episodes):
            obs = self._env.reset()
            done, steps, episode_reward = False, 0, 0

            batch = defaultdict(list)
            reward_list = []
            while steps < time_steps and not done:

                action = self._calculate_action(obs)

                assert isinstance(action, int)

                obs, rewards, done, info = self._env.step(action=action)

                stats = self._agent.ppo_trainer.step(
                    queries=[torch.tensor(obs).flatten().to("cuda")],
                    responses=[torch.tensor([action]).to("cuda")],
                    scores=[torch.tensor([rewards]).to("cuda")],
                )

                steps += 1
                episode_reward += rewards

            self._save_checkpoint()
        self._env._write_av_reward_per_episode()  # noqa
        self.save()
        self._env.close()
        super().learn()

        # save agent
        self.save()

        self._plot_av_reward_per_episode(learning_session=True)

    def _calculate_action(self, obs: np.ndarray):
        action, prompt, reasoning = self._calculate_action_info(obs)

        return action

    def _calculate_action_info(self, obs: np.ndarray) -> tuple[int, str | None, str | None]:
        prev_env_state = self.env_history[-1]
        env_state = EnvironmentState(self._env, prev_env_state=prev_env_state)

        action, prompt, reasoning = self._agent.predict(env_state, self.env_history)
        env_state.action_id = action
        self.env_history.append(env_state)

        return action, None, None

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

                assert isinstance(action, int)

                obs, rewards, done, info = self._env.step(action=action)
                steps += 1
                rew += rewards

        self._env._write_av_reward_per_episode()  # noqa
        self._env.close()
        super().evaluate()

    def _get_latest_checkpoint(self):
        pass

    @classmethod
    def load(cls, path):
        pass

    def save(self):
        self._save_checkpoint()

    def export(self) -> None:
        return None
