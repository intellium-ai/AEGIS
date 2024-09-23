from logging import Logger
import logging
from pathlib import Path
from typing import Any
import os
import datetime
import numpy as np
import json

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical
from torch_geometric.nn import GATConv, LayerNorm, global_add_pool, JumpingKnowledge, global_mean_pool
from torch_geometric.data.batch import Batch

from primaite import getLogger
from primaite.agents.aegis.modules.ge import GraphEmbedding
from primaite.agents.agent_abc import AgentSessionABC
from primaite.agents.utils import from_networkx, prepare_graph
from primaite.common.enums import AgentFramework, AgentIdentifier
from primaite.environment.primaite_env import Primaite

logging.getLogger().setLevel(logging.INFO)

# Maximum number of nodes in any input graph
N_MAX = 50

class Actor(nn.Module):
    def __init__(self, hidden_dim, n_gat_layers: int = 3, device: str = "cuda:0"):
        self.hidden_dim = hidden_dim
        self.n_gat_layers = n_gat_layers
        self.device = device
        super().__init__()
        self.gat = GATConv(in_channels=6, out_channels=hidden_dim)
        self.inner_gat = GATConv(in_channels=hidden_dim, out_channels=hidden_dim)
        self.layer_norm = LayerNorm(hidden_dim)
        self.jumping_knowledge = JumpingKnowledge(mode="max")
        self.linear = nn.Linear(in_features=hidden_dim, out_features=N_MAX + 1 + 4 + 4 + 3)
        self.device = device
        self.n_gat_layers = n_gat_layers

        self.init_weights()

    def forward(self, graph):
        batch = Batch.from_data_list([graph]).to(self.device)

        x = batch.x.to(self.device)
        edge_index = batch.edge_index.to(self.device)

        x = self.gat(x, edge_index)
        if self.n_gat_layers > 1:
            xs = []
            for _ in range(self.n_gat_layers):
                x = self.inner_gat(x, edge_index)
                x = self.layer_norm(x)
                xs.append(x)
            x = self.jumping_knowledge(xs)
        else:
            x = x

        x = global_mean_pool(x=x, batch=batch.batch.to(self.device))
        x = F.relu(x)
        x = self.linear(x)

        # Want to get output shape (N_MAX+1, 4, 4, 3) from a length N_MAX + 12 long vector
        a1_logits = x[:, : N_MAX + 1]
        a2_logits = x[:, N_MAX + 1 : N_MAX + 1 + 4]
        a3_logits = x[:, N_MAX + 1 + 4 : N_MAX + 1 + 4 + 4]
        a4_logits = x[:, N_MAX + 1 + 4 + 4 : N_MAX + 1 + 4 + 4 + 3]

        return (a1_logits, a2_logits, a3_logits, a4_logits)

    def init_weights(self):
        torch.nn.init.normal_(self.linear.weight, 0.01, 0.1)
        torch.nn.init.zeros_(self.linear.bias)

    def save(self, path: str) -> None:
        # Save init arguments
        init_kwargs = {
            "n_gat_layers": self.n_gat_layers,
            "hidden_dim": self.hidden_dim,
            "device": self.device,
        }

        json.dump(init_kwargs, open(os.path.join(path, "actor_init_kwargs.json"), "w"))

        # Save model
        torch.save(self.state_dict(), os.path.join(path, "actor.pt"))

    @classmethod
    def load(cls, path: str):
        init_kwargs = json.load(open(os.path.join(path, "actor_init_kwargs.json")))
        ge = cls(**init_kwargs)
        ge.load_state_dict(torch.load(os.path.join(path, "actor.pt")))
        return ge.to(ge.device)


class Critic(nn.Module):
    def __init__(self, hidden_dim, n_gat_layers: int = 3, device: str = "cuda:0"):
        self.hidden_dim = hidden_dim
        self.n_gat_layers = n_gat_layers
        self.device = device

        super().__init__()
        self.gat = GATConv(in_channels=6, out_channels=hidden_dim)
        self.inner_gat = GATConv(in_channels=hidden_dim, out_channels=hidden_dim)
        self.layer_norm = LayerNorm(hidden_dim)
        self.jumping_knowledge = JumpingKnowledge(mode="max")
        self.linear = nn.Linear(in_features=hidden_dim, out_features=1)
        self.device = device
        self.n_gat_layers = n_gat_layers

        self.init_weights()

    def forward(self, graph):
        batch = Batch.from_data_list([graph]).to(self.device)

        x = batch.x.to(self.device)
        edge_index = batch.edge_index.to(self.device)

        x = self.gat(x, edge_index)
        if self.n_gat_layers > 1:
            xs = []
            for _ in range(self.n_gat_layers):
                x = self.inner_gat(x, edge_index)
                x = self.layer_norm(x)
                xs.append(x)
            x = self.jumping_knowledge(xs)
        else:
            x = x

        x = global_mean_pool(x=x, batch=batch.batch.to(self.device))
        x = F.relu(x)
        x = self.linear(x)
        return x

    def init_weights(self):
        torch.nn.init.normal_(self.linear.weight, 1.0, 0.1)
        torch.nn.init.zeros_(self.linear.bias)

    def save(self, path: str) -> None:
        # Save init arguments
        init_kwargs = {
            "n_gat_layers": self.n_gat_layers,
            "hidden_dim": self.hidden_dim,
            "device": self.device,
        }

        json.dump(init_kwargs, open(os.path.join(path, "critic_init_kwargs.json"), "w"))

        # Save model
        torch.save(self.state_dict(), os.path.join(path, "critic.pt"))

    @classmethod
    def load(cls, path: str):
        init_kwargs = json.load(open(os.path.join(path, "critic_init_kwargs.json")))
        ge = cls(**init_kwargs)
        ge.load_state_dict(torch.load(os.path.join(path, "critic.pt")))
        return ge.to(ge.device)


class GNNAgent(AgentSessionABC):
    def __init__(
        self,
        training_config_path,
        lay_down_config_path,
        hidden_dim: int = 128,
        gat_layers: int = 3,
        gamma: float = 0.99,
        actor_lr: float = 0.001,
        critic_lr: float = 0.01,
        device: str = "cuda:0",
    ):
        super().__init__(training_config_path, lay_down_config_path)
        assert self._training_config.agent_framework == AgentFramework.CUSTOM
        assert self._training_config.agent_identifier == AgentIdentifier.GNN

        self.gamma = gamma
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        self.hidden_dim = hidden_dim
        self.gat_layers = gat_layers

        self.device = device

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

        self.actor = Actor(hidden_dim=self.hidden_dim, n_gat_layers=self.gat_layers, device=self.device).to(self.device)

        self.critic = Critic(hidden_dim=self.hidden_dim, n_gat_layers=self.gat_layers, device=self.device).to(
            self.device
        )

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        self.writer = SummaryWriter(
            flush_secs=15,
            log_dir=f'./gnn_agent_runs/{datetime.datetime.now().strftime("%d-%H:%M:%S")}__hd{self.hidden_dim}_l{self.gat_layers}_g{self.gamma}_a{self.actor_lr}_c{self.critic_lr}',
        )

        # Keep track of env history
        super()._setup()

        self._can_learn = True
        self._can_evaluate = True

        self.action_list_to_int = {tuple(value): key for key, value in self._env.action_dict.items()}

    def _save_checkpoint(self) -> None:
        pass

    def create_graph(self, obs):
        graph = prepare_graph(self._env.network)
        state = from_networkx(graph)
        state.x = torch.tensor(obs[: self._env.num_nodes, 1:], dtype=torch.float32).to(self.device)
        return state

    def decode_logits(self, components_logits: tuple[torch.tensor]) -> tuple[tuple[int], torch.Tensor]:
        action, logprob = [], torch.tensor([0], device=self.device, dtype=torch.float32)

        for idx, logits in enumerate(components_logits):
            logits = logits.squeeze()

            # Find invalid actions and mask out
            valid_component_indicies = set([act[idx] for act in self._env.action_dict.values() if act[:idx] == action])
            invalid_component_indicies = set(range(logits.shape[0])).difference(valid_component_indicies)

            if len(invalid_component_indicies) > 0:
                invalid_component_indicies = torch.tensor(list(invalid_component_indicies))
                logits[invalid_component_indicies] = -torch.inf

            # Generate distribution
            dist = Categorical(logits=logits)
            sampled_index = dist.sample()
            sampled_component = torch.tensor([act[idx] for act in self._env.action_dict.values()]).unique()[sampled_index]

            # Add component to action and add logprob to total
            action.append(sampled_component.item())
            logprob = torch.nn.functional.log_softmax(logits, dim=0)[sampled_index]

            # If a 0 is sampled at index 0, return the default action
            if idx == 0 and sampled_component == 0:
                return (0,0,0,0), logprob
            
        return tuple(action), logprob

    def evaluate(self, time_steps: int = 128, episodes: int = 128, **kwargs):
        self.is_eval = True

        global_step = 0
        reward_per_ep = []

        for ep in range(episodes):
            obs = self._env.reset()
            done, step = False, 0

            episode_rewards = []
            while step < time_steps and not done:
                # Format Data and pass through actor
                with torch.no_grad():
                    action_components_logits = self.actor(self.create_graph(obs))

                # Sample action and calculate probabilites
                sampled_action, _ = self.decode_logits(action_components_logits)

                # Check if action valid:
                if sampled_action in self._env.action_dict.values():
                    action = self.action_list_to_int[tuple(sampled_action)]
                else:
                    action = 0

                new_obs, reward, done, _ = self._env.step(action)

                episode_rewards.append(reward)

                step += 1
                global_step += 1
                obs = new_obs

            reward_per_ep.append(np.sum(episode_rewards))

        return reward_per_ep

    def learn(self, time_steps: int = 128, episodes: int = 128, **kwargs):
        actor_lr_scheduler = CosineAnnealingLR(self.actor_optimizer, T_max=episodes)
        criticr_lr_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max=episodes)

        self.is_eval = False

        global_step = 0
        reward_per_ep = []

        for ep in range(episodes):
            obs = self._env.reset()
            done, step = False, 0

            episode_rewards = []

            while step < time_steps and not done:
                # Format Data and pass through actor
                action_component_logits = self.actor(self.create_graph(obs))

                # Sample action and calculate probabilites
                sampled_action, action_logprob = self.decode_logits(action_component_logits)
                    
                # Check if action valid:
                if sampled_action in self._env.action_dict.values():
                    action = self.action_list_to_int[tuple(sampled_action)]
                else:
                    action = 0

                new_obs, reward, done, _ = self._env.step(action)

                state_value = self.critic(self.create_graph(obs))[0]

                with torch.no_grad():
                    new_state_value = self.critic(self.create_graph(new_obs))[0]

                advantage = reward + self.gamma * new_state_value - state_value

                actor_loss = -action_logprob * advantage.detach()

                critic_loss = torch.square(advantage)
                critic_loss.backward()

                self.critic_optimizer.step()
                self.critic_optimizer.zero_grad()

                self.writer.add_scalar("step/critic_loss", critic_loss.detach().cpu().item(), global_step=global_step)

                actor_loss.backward()
                self.actor_optimizer.step()
                self.actor_optimizer.zero_grad()

                self.writer.add_scalar("step/actor_loss", actor_loss.detach().cpu().item(), global_step=global_step)
                self.writer.add_scalar("step/reward", reward, global_step=global_step)

                episode_rewards.append(reward)
                step += 1
                global_step += 1
                obs = new_obs

            reward_per_ep.append(np.sum(episode_rewards))
            self.writer.add_scalar("episode/reward_avg", np.mean(episode_rewards), global_step=ep + 1)

            actor_lr_scheduler.step()
            criticr_lr_scheduler.step()

        np.save(os.path.join(self.writer.log_dir, "reward_per_ep.npy"), reward_per_ep)
        # self._env.close()
        # super().learn()

        return reward_per_ep

    def _calculate_action(self, obs: np.ndarray) -> int:
        return  # super()._calculate_action(obs)

    def _get_latest_checkpoint(self) -> None:
        return  # super()._get_latest_checkpoint()

    @classmethod
    def load(cls, path):
        return None

    def save(self):
        return None

    def export(self) -> None:
        return None
