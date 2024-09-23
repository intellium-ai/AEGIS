from pathlib import Path
from primaite.agents.agent_abc import AgentSessionABC
from primaite.agents.git_agent import GITAgent
from primaite.agents.gnn_agent import GNNAgent
from primaite.agents.rllib import RLlibAgent
from primaite.agents.sb3 import SB3Agent
from primaite.agents.vanilla_llm import LLMAgent, TrainableLLMAgent
from primaite.agents.simple import RandomAgent, DoNothingNodeAgent, PickACardAnyCardAgent

agent_map = {
    "git": GITAgent,
    "gnn": GNNAgent,
    "rllib": RLlibAgent,
    "sb3": SB3Agent,
    "llm": LLMAgent,
    "trainable_llm": TrainableLLMAgent,
    "random": RandomAgent,
    "do_nothing": DoNothingNodeAgent,
    "random_card": PickACardAnyCardAgent
}

class Agent:
    def __init__(
        self,
        agent_class: str | type[AgentSessionABC],
        training_config_path: str | Path,
        lay_down_config_path: str | Path
    ):
        self.agent: AgentSessionABC = self.__get_agent(agent_class, training_config_path, lay_down_config_path)

    @staticmethod
    def __get_agent(agent_class: str | type[AgentSessionABC], training_config_path: str | Path, lay_down_config_path: str | Path) -> AgentSessionABC:
        if isinstance(agent_class, str):
            if agent_class in agent_map:
                return agent_map[agent_class](training_config_path=training_config_path, lay_down_config_path=lay_down_config_path)
            raise ValueError(f"Unknown agent type: {agent_class}")
        
        for cls in agent_map.values():
            if agent_class == cls:
                return cls(training_config_path=training_config_path, lay_down_config_path=lay_down_config_path)
        raise ValueError(f"Unknown agent class: {agent_class}")