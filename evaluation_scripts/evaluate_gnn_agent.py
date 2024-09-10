from primaite.agents.gnn_agent import GNNAgent
import os
import pandas as pd

lay_down_configs = os.listdir("../data/datasets/GLLM_Dataset/")
results = pd.DataFrame(columns=["lay_down_config", "avg_reward"])
for lay_down in lay_down_configs:
    if lay_down.endswith('.yaml'):
        lay_down_eval_rewards = []
        agent = GNNAgent(
                training_config_path="../agents/training_configs/gnn.yaml",
                lay_down_config_path=f"../data/datasets/GLLM_Dataset/{lay_down}",
            )
        avg_ep_reward = agent.evaluate()
        lay_down_eval_rewards.append(avg_ep_reward)