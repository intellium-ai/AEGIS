from datetime import datetime
from primaite.agents.rllib import RLlibAgent
from primaite.agents.sb3 import SB3Agent
from stable_baselines3.a2c import A2C
import pandas as pd
from stable_baselines3.ppo import PPO
from torch.utils.tensorboard import SummaryWriter
import os
import logging


corpus_sizes = ["5", "10", "20", "40"]
eval_results_df = pd.DataFrame(columns=["corpus_size", "run", "episode", "sum_reward"])
train_results_df = pd.DataFrame(columns=["corpus_size", "run", "episode", "sum_reward"])

run_timestamp = datetime.now().strftime("%H-%M-%d-%m")

# For each corpus size, train and evaluate model on same laydown for 128 episodes and 128 steps
for corpus_size in corpus_sizes:
    for run in range(1):
        writer = SummaryWriter(flush_secs=15, log_dir=f"./runs/{run_timestamp}/training_{corpus_size}_nodes")
        laydown_path = "/srv/aegis-evaluation-data/eval_dataset/corpus_size_" + corpus_size + "/"
        save_path = f"/srv/aegis-trained-agents/{corpus_size}_sb3_{run_timestamp}/"
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        agent = SB3Agent(
            training_config_path="../agents/training_configs/sb3.yaml",
            lay_down_config_path=laydown_path + "0.yaml",
        )

        sum_ep_rewards = agent.learn()
        for episode, reward in enumerate(sum_ep_rewards):
            writer.add_scalar("train_episode/reward", reward, global_step=episode + 1)
            train_results_df.loc[len(train_results_df)] = {
                "run": run + 1,
                "corpus_size": corpus_size,
                "episode": episode + 1,
                "sum_reward": reward,
            }

        current_checkpoint_path = agent._agent.save(save_path + "model")
        print(f"Finished training of {corpus_size}, saved in {current_checkpoint_path}")
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

eval_results_df.to_csv("./results/sb3_eval_results.csv", index=False)
train_results_df.to_csv("./results/sb3_train_results.csv", index=False)
