from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from gym.spaces import Space
from stable_baselines3.common.base_class import BaseAlgorithm, BaseAlgorithmSelf
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
from stable_baselines3.common.utils import get_device
from torch import Tensor, device


class RandomPolicy(BasePolicy):

    def __init__(self, *args, squash_output: bool = False, **kwargs):
        super().__init__(*args, squash_output=squash_output, **kwargs)

    def _predict(self, observation: Tensor, deterministic: bool = False) -> Tensor:
        # assigning to a variable here to help with type hints
        action_space: Space = self.action_space
        return Tensor([action_space.sample()])


class RandomAgent(BaseAlgorithm):

    def __init__(
        self,
        policy: type[BasePolicy],
        env: GymEnv,
        learning_rate: float | Callable[[float], float] = 1,
        policy_kwargs: Dict[str, Any] | None = None,
        tensorboard_log: str | None = None,
        verbose: int = 0,
        device: device | str = "auto",
        support_multi_env: bool = False,
        create_eval_env: bool = False,
        monitor_wrapper: bool = True,
        seed: int | None = None,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        supported_action_spaces: Tuple | None = None,
    ):

        self.device = get_device(device=device)

        self.action_space = env.action_space
        self.observation_space = env.observation_space

        self.policy = policy(
            observation_space=self.observation_space,
            action_space=self.action_space,
        )

    def _setup_model(self) -> None:
        return None

    def learn(
        self: BaseAlgorithmSelf,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 100,
        tb_log_name: str = "run",
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> BaseAlgorithmSelf:
        return self

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        random_action = self.policy.predict(
            observation=observation,
            state=state,
            episode_start=episode_start,
            deterministic=deterministic,
        )

        return random_action
