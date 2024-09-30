from datetime import datetime
from primaite.agents.gnn_agent import GNNAgent
import os
import logging
import random
import pandas as pd

logging.disable(logging.CRITICAL)

# Specifying training and evalution laydowns folder
evaluation_laydowns = "/srv/aegis-evaluation-data/eval_dataset/"

# Setting up trained model storage path
run_timestamp = datetime.now().strftime("%H-%M-%d-%m")

# Evaluate on 5, 10, 20 and 40 node networks
corpus_sizes = ["5", "10", "20", "40"]
eval_results_df = pd.DataFrame(columns=["corpus_size", "run", "episode", "sum_reward"])

laydowns = os.listdir(evaluation_laydowns)

first = True

# Evaluate the model on networks of size 5, 10, 20 and 40 nodes respectively for 128 episodes of 128 steps.
for corpus_size in corpus_sizes:
    agent = GNNAgent(
        training_config_path="../agents/training_configs/gnn.yaml",
        lay_down_config_path=evaluation_laydowns + corpus_size + "_nodes.yaml",
    )
    agent.learn()
    sum_ep_rewards = agent.evaluate()
    for episode, reward in enumerate(sum_ep_rewards):
        eval_results_df.loc[len(eval_results_df)] = {
            "corpus_size": corpus_size,
            "episode": episode + 1,
            "sum_reward": reward,
        }
    print(f"Finished evaluating corpus size {corpus_size}", flush=True)
    eval_results_df.to_csv("./results/fair_gnn_eval_results.csv", index=False)
