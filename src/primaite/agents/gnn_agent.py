from logging import Logger
from pathlib import Path
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.distributions import Categorical
import torch.optim as optim
from primaite import getLogger
from primaite.agents.aegis.modules.ge import GraphEmbedding
from torch_geometric.data.batch import Batch
from primaite.agents.agent_abc import AgentSessionABC
from primaite.agents.utils import from_networkx, prepare_graph
from primaite.common.enums import AgentFramework, AgentIdentifier
from primaite.environment.primaite_env import Primaite

_LOGGER: Logger = getLogger(__name__)

device = "cuda:0"
learning_rate = 0.0005

class GNNAgent(AgentSessionABC):
    def __init__(self, training_config_path, lay_down_config_path):
        super().__init__(training_config_path, lay_down_config_path)
        assert self._training_config.agent_framework == AgentFramework.CUSTOM
        assert self._training_config.agent_identifier == AgentIdentifier.GNN
        self.roll_out = []
        self._setup()

    def _setup(self):

        if not isinstance(self.session_path, Path):
            self.session_path = Path(self.session_path)

        self._env = Primaite(
            training_config_path=self._training_config_path,
            lay_down_config_path=self._lay_down_config_path,
            session_path=self.session_path,
            timestamp_str=self.timestamp_str,
        )
    
        self._agent = GraphEmbedding(
            in_channels=6, output_dim=721, n_tokens=1, hidden_dim=128, device=device
        ).to(device)


        # Keep track of env history
        super()._setup()

        self._can_learn = True
        self._can_evaluate = True

    def _save_checkpoint(self) -> None:
        pass

    def create_graph(self, obs):
        graph = prepare_graph(self._env.network)
        state = from_networkx(graph)
        state.x = torch.tensor(obs[:self._env.num_nodes, 1:], dtype=torch.float32).to(device)
        return state

    def _calculate_action(self, obs) -> int:
        data = self.create_graph(obs)
        batch = Batch.from_data_list([data]).to(device)
        a_prob = self._agent(batch)
        
        a_distrib = Categorical(logits=a_prob)
        action = a_distrib.sample().item()
        return int(action)

    def evaluate(
        self,
        **kwargs: Any,
    ) -> None:
        """
        Evaluate the agent.

        :param kwargs: Any agent-specific key-word args to be passed.
        """
        avg_ep_rewards = []
        time_steps = self._training_config.num_eval_steps
        episodes = self._training_config.num_eval_episodes
        self._env.set_as_eval()
        self.is_eval = True

        for _ in range(episodes):

            obs = self._env.reset()
            done, steps, rew = False, 0, 0

            while steps < time_steps and not done:
                print(steps)
                action = self._calculate_action(obs)
                if int(action) >= int(self._env.action_space.n):
                    print('Illegal action produced. Selecting 0 and setting reward to high negative value')
                    rewards = np.float32(-0.025)
                    obs, rewards, done, _ = self._env.step(action=0)
                else:
                    obs, rewards, done, _ = self._env.step(action=int(action))
                steps += 1
                rew += rewards

        # self._env._write_av_reward_per_episode()  # noqa
        avg_ep_rewards.append(self._env.average_reward)
        self._env.close()
        super().evaluate()
        return avg_ep_rewards
    def put_data(self, data):
        self.roll_out.append(data)
        
    def learn(self, **kwargs):
        time_steps = self._training_config.num_train_steps
        episodes = self._training_config.num_train_episodes
        self.is_eval = False
        losses = []
        mean_rewards = []
        
        self.optimizer = optim.Adam(self._agent.parameters(), lr=learning_rate)
        for ep in range(episodes):
            obs = self._env.reset()
            done, steps, rew = False, 0, 0

            while steps < time_steps and not done:
                data = self.create_graph(obs)
                batch = Batch.from_data_list([data]).to(device)
                a_prob = self._agent(batch).squeeze(0)

                a_distrib = Categorical(logits=a_prob)
                action = a_distrib.sample().item()
                if int(action) >= int(self._env.action_space.n):
                    print(f'Illegal action produced. Selecting 0 and setting reward to high negative value: {int(action)} {int(self._env.action_space.n)}')
                    rewards = np.float32(-0.025)
                    obs, rewards, done, _ = self._env.step(action=0)
                else:
                    print(f'Legal action produced. {int(action)} {int(self._env.action_space.n)}')
                    obs, rewards, done, _ = self._env.step(action=int(action))
                
                self.put_data((rewards, a_prob[0][action]))

                steps += 1
                rew += rewards
                self._save_checkpoint()

            loss, mean_reward = self.train_net(0.99)
            losses.append(loss)
            mean_rewards.append(mean_reward)
            self._env._write_av_reward_per_episode()
            self.save()

        self._env.close()
        super().learn()

        self._plot_av_reward_per_episode(True)
        
    def train_net(self, gamma):
        R = 0
        G = []
        G_t = 0
        # Whitening baseline
        for r, prob in self.roll_out[::-1]:
            G_t = r + gamma * G_t
            G.append(G_t)

        G = np.array(G)
        G_mean = G.mean()
        G_std = G.std()

        self.optimizer.zero_grad()

        for r, prob in self.roll_out[::-1]:
            R = r + gamma * R
            loss = -prob * ((R - G_mean) / G_std)
            loss.backward()
        self.optimizer.step()
        mean_reward = np.mean([rew[0] for rew in self.roll_out])
        self.roll_out = []

        return loss.cpu().detach().numpy(), mean_reward
    
    def _save_training_fig(self, losses, mean_rewards):
        plt.plot(losses, label="Loss")
        # plt.plot(mean_reward, label='Avg Reward')
        plt.xlabel("Episode #")
        plt.ylabel("Loss")
        plt.savefig("./loss.png")

        plt.close()
        plt.plot(mean_rewards, label="Avg Reward")
        plt.xlabel("Episode #")
        plt.ylabel("Avg Reward")
        plt.savefig("./avg_reward.png")
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
