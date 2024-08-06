from logging import Logger
from pathlib import Path
from typing import Any, Tuple
import torch
from torch.distributions import Categorical
from primaite import getLogger
from primaite.agents.agent_abc import AgentSessionABC
from primaite.common.enums import AgentFramework, AgentIdentifier
from primaite.environment.env_state import EnvironmentState
from primaite.environment.primaite_env import Primaite
from primaite.agents.utils import from_networkx, prepare_graph
from primaite.agents.git.git_policy import GITPolicy
from primaite.action import NodeAction
import matplotlib.pyplot as plt
import logging
_LOGGER: Logger = getLogger(__name__)

device = "cuda:0"

class GITAgent(AgentSessionABC):
    def __init__(self, training_config_path, lay_down_config_path):
        super().__init__(training_config_path, lay_down_config_path)
        assert self._training_config.agent_framework == AgentFramework.CUSTOM
        assert self._training_config.agent_identifier == AgentIdentifier.GIT
        self._setup()

    def _setup(self):
        # you need to find out how to get the probability of each token INTEGER that was produced within the action range, convert this to a tensor and then apply the loss.backward() on this, preceeding a optimizer.backward() pass to update the model weights. Discuss with stefan - will this work?
        if not isinstance(self.session_path, Path):
            self.session_path = Path(self.session_path)

        self._env = Primaite(
            training_config_path=self._training_config_path,
            lay_down_config_path=self._lay_down_config_path,
            session_path=self.session_path,
            timestamp_str=self.timestamp_str,
        )

        self._agent = GITPolicy(state_space=6, action_space=100, hidden_dim=128, ge_learning_rate=0.0001, llm_device='cuda:0', ap_device='cuda:1', ge_device='cuda:0')
        
        logging.info(self._agent)

        # Keep track of env history
        self.env_history = [EnvironmentState(self._env)]
        super()._setup()

        self._can_learn = True
        self._can_evaluate = True

    def _save_checkpoint(self) -> None:
        pass

    def create_graph(self, obs):
        graph = prepare_graph(self._env.network)
        state = from_networkx(graph)
        state.x = torch.tensor(obs[:9, 1:], dtype=torch.float32).to(device)
        return state

    def _calculate_action(self, obs) -> int:
        data = self.create_graph(obs)
        result = self._agent(data.x, data.edge_index)
        return result

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

        for _ in range(episodes):

            obs = self._env.reset()
            done, steps, rew = False, 0, 0

            while steps < time_steps and not done:

                result = self._calculate_action(obs)
                action = int(result.indices[0])
                obs, rewards, done, _ = self._env.step(action=action)
                steps += 1
                rew += rewards

        self._env._write_av_reward_per_episode()  # noqa
        self._env.close()
        super().evaluate()

    def learn(self, **kwargs):
        time_steps = self._training_config.num_train_steps
        episodes = self._training_config.num_train_episodes
        self.is_eval = False
        losses = []
        mean_rewards = []
        for ep in range(episodes):
            obs = self._env.reset()
            done, steps, rew = False, 0, 0

            while steps < time_steps and not done:

                data = self.create_graph(obs)
                output = self._agent(data.x, data.edge_index)
                
                # The action output ID is a token ID, so we need to convert it to an integer:
                action = self._agent.llm.tokenizer.decode(output.indices[0])
                
                try:
                    # Validate that the action ID is a valid action integer. If not, fallback to 0.
                    action = int(action)
                    NodeAction.from_id(env=self._env, action_id=action)
                except:
                    action = 0
                    logging.warning('An invalid action id was produced, falling back to 0')
                
                obs, rewards, done, _ = self._env.step(action=action)
                self._agent.put_data((rewards, output.values[0]))

                steps += 1
                rew += rewards
                self._save_checkpoint()
                

            loss, mean_reward = self._agent.train_net(0.99)
            losses.append(loss)
            mean_rewards.append(mean_reward)
            self._save_training_fig(losses=losses, mean_rewards=mean_rewards)
            logging.info(f'Completed episode: {ep} with loss {loss}')
            self._env._write_av_reward_per_episode()
            self.save()

        self._env.close()
        super().learn()
        
        self._plot_av_reward_per_episode(True)
    def _save_training_fig(self, losses, mean_rewards):
        plt.plot(losses, label='Loss')
        #plt.plot(mean_reward, label='Avg Reward')
        plt.xlabel('Episode #')
        plt.ylabel('Loss')
        plt.savefig('./loss.png')
        
        plt.close()
        plt.plot(mean_rewards, label='Avg Reward')
        plt.xlabel('Episode #')
        plt.ylabel('Avg Reward')
        plt.savefig('./avg_reward.png')
        plt.close()
    def _get_latest_checkpoint(self):
        pass

    @classmethod
    def load(cls, path):
        pass

    def save(self):
        return None

    def export(self) -> None:
        return None
