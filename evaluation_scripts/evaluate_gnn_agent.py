from primaite.agents.gnn_agent import GNNAgent
import os

lay_down_configs = os.listdir("/data/laydown_configs/")
model_folder_path = '/'
lay_down_eval_rewards = []
for lay_down in lay_down_configs:

    agent = GNNAgent(
        training_config_path="./agents/training_configs/gnn.yaml",
        lay_down_config_path=f"./data/laydown_configs/{lay_down}",
    )
    agent._agent = agent._agent.load(model_folder_path)

    avg_ep_rewards = agent.evaluate()
    lay_down_eval_rewards.append(avg_ep_rewards)
