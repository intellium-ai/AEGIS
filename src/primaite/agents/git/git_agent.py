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
from primaite.agents.git.prompts import LLM_PROMPT
from primaite.agents.llm.utils import get_obs_act_history_str, obs_diff
from primaite.environment import EnvironmentState
import matplotlib.pyplot as plt
from primaite.agents.llm.prompting import AgentNodeAction
import logging
_LOGGER: Logger = getLogger(__name__)

device = "cuda:0"

class GITAgent(AgentSessionABC):
    def __init__(self, training_config_path, lay_down_config_path):
        super().__init__(training_config_path, lay_down_config_path)
        assert self._training_config.agent_framework == AgentFramework.CUSTOM
        assert self._training_config.agent_identifier == AgentIdentifier.GIT
        
        self._setup()
        
        # Set the node and service maps for prompt - llm - action conversion
        self.node_mapping = self._env.nodes
        self.service_mapping = {key + 1: value for key, value in enumerate(self._env.services_list)}
        
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

    def _calculate_action(self, obs) -> Tuple[int, torch.Tensor]:
        
        data = self.create_graph(obs)
        prompt = self._build_prompt()
        llm_action_text, probs = self._agent(data.x, data.edge_index, prompt)

        print(f'LLM Generated Text: {llm_action_text}')    
        
        # try:
            # Validate that the action ID is a valid action integer. If not, fallback to 0.
        action = self.get_node_action(text_output=llm_action_text).to_node_action(env=self._env).action_id
        # except:
        #     action = 0
        #     logging.warning('An invalid action id was produced, falling back to 0')
        
        return action, probs

    def _build_prompt(self) -> str:

        # Make the service name just TCP or UDP etc, then find the mapping between nodes and what service they have.
        nodes_table = self.env_history[0].nodes_table.rename(columns={'TCP Service State': 'TCP', 'TCP_SQL Service State': 'TCP_SQL', 'UDP Service State': 'UDP'})
        node_services = {}
        for idx in range(len(nodes_table)):
            node = nodes_table.iloc[idx, :]
            node_services[node['Name']] = []
            
            if node['TCP'] != '-':
                # I hate this
                node_services[node['Name']].append([k for k, v in self.service_mapping.items() if v == 'TCP'][0])
            if node['TCP_SQL'] != '-':
                node_services[node['Name']].append([k for k, v in self.service_mapping.items() if v == 'TCP_SQL'][0])
            if node['UDP'] != '-':
                node_services[node['Name']].append([k for k, v in self.service_mapping.items() if v == 'UDP'][0])
        
        # Get stringified node map, service map, node services, observation and action history and current observation difference for prompt
        node_str = "\n".join(f"{key}: {value}" for key, value in self.node_mapping.items()) # ID: NODE_NAME
        services_str = '\n'.join(f'{i}: {service}' for i, service in enumerate(self.service_mapping))
        node_services_str = "\n".join(f"{key}: {', '.join(str(val) for val in value) if value else 'NONE'}" for key, value in node_services.items())
        obs_act_history_str = get_obs_act_history_str(self.env_history, env=self._env)
        
        if not obs_act_history_str:
            obs_act_history_str = 'No observation history yet...'
        obs_diff_str = obs_diff(EnvironmentState(env=self._env))

        # Build and return prompt
        return LLM_PROMPT.format(node_ids=node_str, services=services_str, node_services=node_services_str, obs_act_history=obs_act_history_str, current_obs_diff=obs_diff_str)
    
    def get_node_action(self, text_output) -> AgentNodeAction:
        splits = text_output.split('.')

        node = self.node_mapping[int(splits[0])]
        match int(splits[1]):
            case 1:
                property_action = 'TURN_ON'
                node_property = 'HARDWARE'
            case 2:
                property_action = 'TURN_OFF'
                node_property = 'HARDWARE'
            case 3:
                property_action = 'RESET'
                node_property = 'HARDWARE'
            case 4:
                property_action = 'PATCH'
                node_property = 'SOFTWARE'
            case 5:
                property_action = 'PATCH'
                node_property = 'SERVICE'

        # It's a service patch, so look for another integer
        if int(splits[1]) == 5:
            service_name = self.service_mapping[int(splits[2])]
        else:
            service_name = 'NONE'
        return AgentNodeAction(node_name=node, node_property=node_property, property_action=property_action, service_name=service_name)
    
    
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

                action, _ = self._calculate_action(obs)
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

                action, probs = self._calculate_action(obs)
                
                obs, rewards, done, _ = self._env.step(action=action)
                self._agent.put_data((rewards, probs))

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
