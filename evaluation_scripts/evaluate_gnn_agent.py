from primaite.agents.gnn_agent import GNNAgent
import os
import pandas as pd

lay_down_configs = os.listdir("../data/laydown_configs/")
n_episodes = 5
results = pd.DataFrame(columns=["lay_down_config", "avg_reward"])
for lay_down in lay_down_configs:
    print(lay_down)
    lay_down_eval_rewards = []
    agent = GNNAgent(
            training_config_path="../agents/training_configs/gnn.yaml",
            lay_down_config_path=f"../data/laydown_configs/{lay_down}",
        )
    for episode in range(n_episodes):
        avg_ep_reward = agent.evaluate()
        lay_down_eval_rewards.append(avg_ep_reward)
