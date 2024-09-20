from primaite.agents.vanilla_llm.agent import LLMAgent
import os
import pandas as pd
import logging
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

logging.disable(logging.CRITICAL)
corpus_sizes = ["5", "10", "20", "40"]
results_df = pd.DataFrame(columns=["corpus_size", "episode", "avg_reward"])
model_folder_path = "/srv/aegis-trained-agents/"  # Specify GNN model folder path here
run_timestamp = datetime.now().strftime("%H-%M-%d-%m")

for corpus_size in corpus_sizes:
    writer = SummaryWriter(flush_secs=15, log_dir=f"./runs/{run_timestamp}/{corpus_size}_nodes")
    laydown_path = "/srv/aegis-evaluation-data/eval_dataset/corpus_size_" + corpus_size + "/"
    agent = LLMAgent(
        training_config_path="../agents/training_configs/llm.yaml",
        lay_down_config_path=laydown_path + "0.yaml",
        base_url="http://192.168.0.8:23333",
    )

    avg_ep_rewards = agent.evaluate()
    for episode, reward in enumerate(avg_ep_rewards):
        writer.add_scalar("episode/reward", reward, global_step=episode + 1)
    for idx, value in enumerate(avg_ep_rewards):
        results_df.loc[len(results_df)] = {"corpus_size": corpus_size, "episode": idx, "avg_reward": value}
results_df.to_csv("gnn_results.csv", index=False)
