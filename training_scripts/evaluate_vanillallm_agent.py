from primaite.agents.vanilla_llm.agent import LLMAgent
import os
import pandas as pd
import logging
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

logging.disable(logging.CRITICAL)
corpus_sizes = ["5", "10", "20", "40"]
eval_results_df = pd.DataFrame(columns=["corpus_size", "run", "episode", "sum_reward"])
train_results_df = pd.DataFrame(columns=["corpus_size", "run", "episode", "sum_reward"])

run_timestamp = datetime.now().strftime("%H-%M-%d-%m")
for corpus_size in corpus_sizes:
    for run in range(1):
        writer = SummaryWriter(flush_secs=15, log_dir=f"./runs/{run_timestamp}/training_{corpus_size}_nodes")
        laydown_path = "/srv/aegis-evaluation-data/eval_dataset/corpus_size_" + corpus_size + "/"

        agent = LLMAgent(
            training_config_path="../agents/training_configs/llm.yaml",
            lay_down_config_path=laydown_path + "0.yaml",
            base_url="http://192.168.0.71:58089",
        )

        print(f"Evaluating {corpus_size}")

        writer = SummaryWriter(flush_secs=15, log_dir=f"./runs/{run_timestamp}/evaluation_{corpus_size}_nodes")
        sum_ep_rewards = agent.evaluate()
        for episode, reward in enumerate(sum_ep_rewards):
            writer.add_scalar("eval_episode/reward", reward, global_step=episode + 1)
            eval_results_df.loc[len(eval_results_df)] = {
                "run": run + 1,
                "corpus_size": corpus_size,
                "episode": episode + 1,
                "sum_reward": reward,
            }
            eval_results_df.to_csv("vanillallm_eval_results.csv", index=False)
