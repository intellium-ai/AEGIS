from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from primaite.action import NodeAction
from primaite.environment import Primaite
from primaite.network import Network


@dataclass
class ObservedState:
    network: Network
    """This is the network frozen in time at a particular state."""
    changes: list[str] | None = None
    """The changes that have happened between the previous observation and the current one. Changes happen from these causes in the given order: (previous) blue action is taken, then green and red pol/ier is done."""
    action: NodeAction | None = None
    """ This blue action was taken AFTER the observed state."""

    @classmethod
    def from_env(cls, env: Primaite, prev_obs_state: ObservedState | None = None):
        network = Network.from_env(env)
        if prev_obs_state is None:
            return cls(network)
        else:
            changes = network.diff(prev_obs_state.network, limited_view=True, colors=False)
            return cls(network, changes)

    @property
    def changes_str(self) -> str:
        if self.changes is None:
            return ""
        return "\n".join(self.changes)

    @property
    def nodes_table(self) -> pd.DataFrame:
        return self.network.nodes_table(limited_view=True)

    @property
    def traffic_table(self) -> pd.DataFrame:
        return self.network.traffic_table()

    def format(self) -> str:
        obs_str = ""
        nodes = self.network.nodes_table(limited_view=True).to_json(orient="records", indent=2)
        obs_str += "\n\nNode Status:\n" + nodes
        links = self.network.traffic_table().to_json(orient="records", indent=2)
        obs_str += "\n\nTraffic Status:\n" + links + "\n\n"

        return obs_str


def network_connectivity_desc(network: Network) -> str:
    # List nodes
    desc = ""
    nodes = network.active_nodes
    nodes_df = pd.DataFrame({"Name": [n.name for n in nodes], "Type": [n.node_type.name for n in nodes]})
    desc += "\nNodes:\n" + nodes_df.to_json(orient="records", indent=2)
    links = network.links

    links_df = pd.DataFrame(
        {
            "Name": [l.name for l in links.values()],
            "Source Node": [l.source_node_name for l in links.values()],
            "Destination Node": [l.dest_node_name for l in links.values()],
        }
    )
    desc += "\n\nLinks:\n" + links_df.to_json(orient="records", indent=2)

    return desc


def get_obs_act_history_str(obs_history: list[ObservedState], max_history: int = 20) -> str:
    """Builds the observation history string used in LLM agent prompts. Uses the latest N in the history"""
    # Build the history of actions
    obs_act_history = "" if obs_history else "NO OBSERVATION HISTORY"
    history_list = []

    for i, obs in enumerate(obs_history[1:]):

        if obs.changes is not None or obs.action is not None:
            history_list.append(f"\nStep {i}:")

            if obs.changes:
                history_list[-1] += "\n" + obs.changes_str

            if obs.action is not None:
                action_verbose = obs.action.verbose(colored=False)
                history_list[-1] += f"\nAction: {action_verbose}\n"

    obs_act_history += "".join(
        history_list[-max_history:]
    )  # Only show up to the last 20 obs act events to avoid cloggage

    return obs_act_history
