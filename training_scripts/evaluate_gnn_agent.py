from primaite.agents.gnn_agent import GNNAgent
import os
import pandas as pd
import logging

logging.disable(logging.CRITICAL)
corpus_sizes = ["5", "10", "20", "40"]
results_df = pd.DataFrame(columns=["corpus_size", "episode", "avg_reward"])
model_folder_path = "/srv/aegis-trained-agents/"  # Specify GNN model folder path here

for corpus_size in corpus_sizes:
    laydown_path = "/srv/aegis-evaluation-data/eval_dataset/corpus_size_" + corpus_size + "/"
    agent = GNNAgent(
        training_config_path="../agents/training_configs/gnn.yaml",
        lay_down_config_path=laydown_path + "0.yaml",
    )
    agent._agent = agent._agent.load(model_folder_path)

    avg_ep_rewards = agent.evaluate()
    for idx, value in enumerate(avg_ep_rewards):
        results_df.loc[len(results_df)] = {"corpus_size": corpus_size, "episode": idx, "avg_reward": value}
results_df.to_csv("gnn_results.csv", index=False)
