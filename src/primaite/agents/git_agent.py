import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple
import time
import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
from torch_geometric.data.batch import Batch
from torch.utils.tensorboard import SummaryWriter

from primaite.agents.aegis.policies.git_policy import GITPolicy
from primaite.agents.aegis.prompts import LLM_PROMPT, LLM_REASONING_PROMPT
from primaite.agents.agent_abc import AgentSessionABC
from primaite.agents.vanilla_llm.observation import get_obs_act_history_str, ObservedState
from primaite.agents.vanilla_llm.prompting import AgentNodeAction
from primaite.agents.utils import from_networkx, prepare_graph
from primaite.common.enums import AgentFramework, AgentIdentifier
from primaite.environment.primaite_env import Primaite
from primaite.network import Network

logging.getLogger().setLevel(logging.INFO)

device = "cuda:0"


class GITAgent(AgentSessionABC):

    @dataclass
    class ActionInfo(AgentSessionABC.ActionInfo): ...

    def __init__(self, training_config_path, lay_down_config_path, save_path: str = None):
        super().__init__(training_config_path, lay_down_config_path)
        assert self._training_config.agent_framework == AgentFramework.CUSTOM
        assert self._training_config.agent_identifier == AgentIdentifier.GIT

        self._setup(save_path=save_path)

        # Set the node and service maps for prompt - llm - action conversion
        self.node_mapping = self._env.nodes
        self.service_mapping = {key + 1: value for key, value in enumerate(self._env.services_list)}

    def _setup(self, save_path: str = None):
        if not isinstance(self.session_path, Path):
            self.session_path = Path(self.session_path)

        self._env = Primaite(
            training_config_path=self._training_config_path,
            lay_down_config_path=self._lay_down_config_path,
            session_path=self.session_path,
            timestamp_str=self.timestamp_str,
        )

        if save_path is not None:
            self._agent = GITPolicy.load(save_path)
        else:
            self._agent = GITPolicy(
                state_space=6,
                hidden_dim=128,
                n_graph_tokens=20,
                ge_device="cuda:0",
            )

        logging.info(self._agent)

        # Keep track of env history
        self.obs_history = [ObservedState.from_env(self._env)]
        super()._setup()

        self._can_learn = True
        self._can_evaluate = True

        self.optimiser = torch.optim.Adam(self._agent.parameters(), maximize=True, lr=0.01)
        if save_path is not None:
            self.optimiser.load_state_dict(torch.load(os.path.join(save_path, 'git_agent_optim_state.pt')))
        self.writer = SummaryWriter(flush_secs=15)

    def _save_checkpoint(self) -> None:
        pass

    def create_graph(self, obs):
        """Construct a graph from the observation space, using only the node features"""
        graph = prepare_graph(self._env.network)
        state = from_networkx(graph)
        state.x = torch.tensor(obs[: self._env.num_nodes, 1:], dtype=torch.float32, device=device)
        state = Batch.from_data_list([state])
        return state

    def _calculate_action(self, obs) -> Tuple[int, torch.Tensor]:

        graph_batch = self.create_graph(obs)
        action_prompt, reasoning_prompt = self._build_prompt()
        llm_action_text, probs, reasoning_statement, pre_action_input = self._agent(graph_batch, action_prompt, reasoning_prompt)

        # Validate that the action ID is a valid action integer. If not, fallback to 0.
        try:
            action = self.get_node_action(text_output=llm_action_text)
            network = Network.from_env(self._env)
            action = action.to_node_action(network=network).action_id
        except:
            action = 0
            print('Invalid LLM Action')
        return action, torch.sum(torch.log(probs)), pre_action_input

    def calculate_action_info(self, obs: np.ndarray) -> tuple[int, ActionInfo]: ...

    def _build_prompt(self) -> Tuple[str, str]:

        # Make the service name just TCP or UDP etc, then find the mapping between nodes and what service they have.
        nodes_table = self.obs_history[0].nodes_table.rename(
            columns={"TCP Service State": "TCP", "TCP_SQL Service State": "TCP_SQL", "UDP Service State": "UDP"}
        )
        node_services = {}
        for idx in range(len(nodes_table)):
            node = nodes_table.iloc[idx, :]
            node_services[node["Name"]] = []

            if node["TCP"] != "-":
                node_services[node["Name"]].append([k for k, v in self.service_mapping.items() if v == "TCP"][0])
            if node["TCP_SQL"] != "-":
                node_services[node["Name"]].append([k for k, v in self.service_mapping.items() if v == "TCP_SQL"][0])
            if node["UDP"] != "-":
                node_services[node["Name"]].append([k for k, v in self.service_mapping.items() if v == "UDP"][0])

        # Get stringified node map, service map, node services, observation and action history and current observation difference for prompt
        node_str = "\n".join(f"{key}: {value}" for key, value in self.node_mapping.items())  # ID: NODE_NAME
        services_str = "\n".join(f"{i}: {service}" for i, service in enumerate(self.service_mapping))
        node_services_str = "\n".join(
            f"{key}: {', '.join(str(val) for val in value) if value else 'NONE'}"
            for key, value in node_services.items()
        )
        obs_act_history_str = get_obs_act_history_str(self.obs_history)

        if not obs_act_history_str:
            obs_act_history_str = "No observation history yet..."

        curr_state = ObservedState.from_env(env=self._env, prev_obs_state=self.obs_history[-1])

        obs_diff_str = curr_state.changes_str

        # Build and return prompts
        reasoning_prompt = LLM_REASONING_PROMPT.format(
            node_ids=node_str,
            services=services_str,
            node_services=node_services_str,
            obs_act_history=obs_act_history_str,
            current_obs_diff=obs_diff_str,
        )

        # Format all except reasoning statement, which is added later (GIT forward pass) after it is generated
        action_prompt = LLM_PROMPT.format(
            node_ids=node_str,
            services=services_str,
            node_services=node_services_str,
            obs_act_history=obs_act_history_str,
            current_obs_diff=obs_diff_str,
            reasoning_statement="{reasoning_statement}",
        )

        return action_prompt, reasoning_prompt

    def get_node_action(self, text_output) -> AgentNodeAction:
        """Parses the LLM action output into an AgentNodeAction

        Args:
            text_output (str): LLM action output

        Returns:
            AgentNodeAction: The parsed AgentNodeAction
        """

        splits = text_output.split(".")

        if splits[0] in self.node_mapping.keys():
            node = str(self.node_mapping[splits[0]])
        else:
            return AgentNodeAction(node_name="NONE", node_property="NONE", property_action="NONE", service_name="NONE")

        match int(splits[1]):
            case 1:
                property_action = "TURN_ON"
                node_property = "HARDWARE"
                service_name = "NONE"
            case 2:
                property_action = "TURN_OFF"
                node_property = "HARDWARE"
                service_name = "NONE"
            case 3:
                property_action = "RESET"
                node_property = "HARDWARE"
                service_name = "NONE"
            case 4:
                property_action = "PATCH"
                node_property = "SOFTWARE"
                service_name = "NONE"
            case 5:
                property_action = "PATCH"
                node_property = "SERVICE"
                service_name = self.service_mapping[int(splits[2])]
            case _:
                node = "NONE"
                property_action = "NONE"
                node_property = "NONE"
                service_name = "NONE"


        return AgentNodeAction(
            node_name=str(node), node_property=node_property, property_action=property_action, service_name=service_name
        )

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
            state = self._env.reset()
            terminated, steps, rew = False, 0, 0

            episode_rewards = []

            while steps < time_steps and not terminated:
                with torch.no_grad():
                    action, _, _ = self._calculate_action(state)
                state, reward, terminated, _ = self._env.step(action=action)
                steps += 1
                
                episode_rewards.append(reward)

        self._env.close()
        super().evaluate()

    def learn(self, **kwargs):
        discount_factor = kwargs.pop('discount_factor', 0.9)

        max_steps = 50 #self._training_config.num_train_steps
        num_episodes = 20 #self._training_config.num_train_episodes
        self.is_eval = False

        global_step = 0

        pbar = tqdm(total=max_steps, unit='step')
        for episode in range(num_episodes):
            pbar.reset()
            pbar.set_description(desc=f'Episode {episode} Running')

            episode_rewards = []
            episode_pre_action_inps = []
            episode_action_tokens = []
            epsiode_actions = []
            step_times = []

            state = self._env.reset()

            terminated, step = False, 0
            while step  < max_steps and not terminated:

                with torch.no_grad():
                    action, _, pre_action_input = self._calculate_action(state)
                state, reward, terminated, _ = self._env.step(action=action)

                episode_rewards.append(reward)
                episode_pre_action_inps.append(pre_action_input[0])
                episode_action_tokens.append(pre_action_input[1])
                epsiode_actions.append(action)

                #torch.cuda.empty_cache()
                
                pbar.update(1)
                pbar.set_postfix({'Reward': reward})
                self.writer.add_scalar('step/reward', reward, global_step=global_step+step)
                step_times.append(time.time())

                step += 1

            pbar.set_description(desc=f'Episode {episode} Done - Reward Sum: {np.sum(episode_rewards)}')
            self.writer.add_scalar('episode/reward', np.sum(episode_rewards), global_step=episode)
            self.writer.add_histogram('episode/actions', np.array(epsiode_actions), global_step=episode)

            # End of episode
            expected_return = 0
            self.optimiser.zero_grad()

            step_update_values = []

            for t in range(step-1, -1, -1):
                expected_return += episode_rewards[t]

                step_update_val = 0

                for action_token_prob in self._agent.llm.yield_probs_given_tokens(
                    inputs_embeds  = episode_pre_action_inps[t].to(torch.device('cuda:0')), 
                    attention_mask = torch.ones((1, episode_pre_action_inps[t].shape[1]), device='cpu'),
                    token_ids      = episode_action_tokens[t]
                ):
                    update_value = (discount_factor ** t) * expected_return * action_token_prob
                    update_value.backward()

                    step_update_val += update_value.detach().cpu().item()

                step_update_values.append(step_update_val)
                expected_return *= discount_factor
            
            for i, val in enumerate(step_update_values):
                self.writer.add_scalar('step/update_val', val, walltime=step_times[t], global_step=global_step+i)

                
            self.optimiser.step()

            global_step += step

            logging.info(f"Completed episode: {episode}")
            self.writer.add_scalar('episode/update_val', np.sum(step_update_values), global_step=episode)
            self.writer.add_scalar('episode/num_steps', step, global_step=episode)
            self.writer.flush()

        pbar.close()
        self._env.close()
        self.writer.close()

    def _get_latest_checkpoint(self):
        pass

    @classmethod
    def load(cls, path):
        raise NotImplementedError

    def save(self, path: str) -> None:
        self._agent.save(path)
        torch.save(self.optimiser.state_dict(), os.path.join(path, 'git_agent_optim_state.pt'))

    def export(self) -> None:
        return None
