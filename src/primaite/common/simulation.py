from __future__ import annotations

from pathlib import Path

from primaite.agents import AgentSessionABC
from primaite.common.env_state import EnvironmentState
from primaite.primaite_session import PrimaiteSession


class Simulation:

    def __init__(self, session: PrimaiteSession):
        session.setup()
        self.agent: AgentSessionABC = session._agent_session
        self.env = self.agent._env
        self.env.set_as_eval()

        self.reset()

    def reset(self):
        init_state = EnvironmentState(env=self.env, obs=self.env.reset())
        self.history = [init_state]

        # MOVE TO INIT ??
        self.num_steps = self.agent._training_config.num_eval_steps
        self.total_reward = 0

        self.done = False

    @classmethod
    def from_laydown(cls, laydown_path: Path | str, training_path: Path | str):
        session = PrimaiteSession(training_config_path=training_path, lay_down_config_path=laydown_path)
        return cls(session)

    @classmethod
    def from_session_file(cls, session_path: Path | str):
        session = PrimaiteSession(session_path=session_path)
        return cls(session)

    @property
    def latest_state(self):
        return self.history[-1]

    @property
    def curr_step(self):
        return len(self.history) - 1

    @property
    def avg_reward(self):
        if self.curr_step == 0:
            return 0.0
        return round(self.total_reward / self.curr_step, 5)

    def is_done(self) -> bool:
        return self.done

    def step(self):
        """Make a step in the simulation. Returns a list of changes between previous state and next state"""
        if self.done:
            raise StopIteration("Simulation is finished. Cannot run another step.")

        last_state = self.latest_state
        obs = last_state.obs
        action, info = self.agent.calculate_action_info(obs)
        obs, rewards, self.done, _ = self.env.step(action)

        rewards = round(rewards, 5)
        self.total_reward += rewards

        curr_state = EnvironmentState(self.env, obs, rewards, action, info, last_state)
        self.history.append(curr_state)
