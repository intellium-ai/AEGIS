# Displaying the environment state
from __future__ import annotations

from dataclasses import asdict

import streamlit as st
from streamlit import session_state as state

from primaite.action import NodeAction
from primaite.agents.agent_abc import AgentSessionABC
from primaite.environment import Primaite
from primaite.network import Network

ActionInfo = AgentSessionABC.ActionInfo


class EnvironmentState:

    def __init__(
        self,
        env: Primaite,
        obs,
        reward: float | None = None,
        action_id: int | None = None,
        info: ActionInfo | None = None,
        prev_state: EnvironmentState | None = None,
    ):

        self.obs = obs
        self.reward = reward
        self.info = info

        self.network = Network.from_env(env=env)
        if action_id is not None:
            try:
                self.action = NodeAction.from_id(self.network, action_id)
            except:
                self.action = None
        else:
            self.action = None

        self.changes = None
        if prev_state is not None:
            self.changes = self.network.diff(prev_state.network, limited_view=False, colors=True)

    @property
    def network_figure(self):
        return self.network.display_graph()

    @property
    def traffic_table(self):
        return self.network.traffic_table()

    @property
    def nodes_table(self):
        return self.network.nodes_table(limited_view=False)

    @property
    def info_str(self) -> str:

        if self.info is None:
            return ""
        return self.info.reasoning
