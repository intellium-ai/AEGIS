from datetime import datetime
from primaite.agents.gnn_agent import GNNAgent
import os
import logging
import random
import pandas as pd

logging.disable(logging.CRITICAL)

# Specifying training and evalution laydowns folder
training_laydowns = "/srv/aegis-training-data/laydowns/group_1/laydowns/"
evaluation_laydowns = "/srv/aegis-evaluation-data/eval_dataset/"

# Setting up trained model storage path
run_timestamp = datetime.now().strftime("%H-%M-%d-%m")
save_path = f"/srv/aegis-trained-agents/gnn_{run_timestamp}/"

# Evaluate on 5, 10, 20 and 40 node networks
corpus_sizes = ["5", "10", "20", "40"]
eval_results_df = pd.DataFrame(columns=["corpus_size", "run", "episode", "sum_reward"])

laydowns = os.listdir(training_laydowns)
random.shuffle(laydowns)

if not os.path.exists(save_path):
    os.mkdir(save_path)
first = True

# Train on all laydowns for 1 episode each
for idx, laydown in enumerate(laydowns):
    if first:
        agent = GNNAgent(
            training_config_path="../agents/training_configs/gnn.yaml",
            lay_down_config_path=training_laydowns + laydown,
        )
    else:
        agent.reset_env(laydown_config_path=training_laydowns + laydown)
        agent.actor = agent.actor.load(save_path)
        agent.critic = agent.critic.load(save_path)

    agent.learn(episodes=8)
    agent.actor.save(save_path)
    agent.critic.save(save_path)
    first = False
    print(f"Finished with laydown {laydown} {idx} of {len(laydowns)}", flush=True)

# Evaluate the model on networks of size 5, 10, 20 and 40 nodes respectively for 128 episodes of 128 steps.
for corpus_size in corpus_sizes:
    agent.reset_env(
        laydown_config_path=evaluation_laydowns + corpus_size + "_nodes.yaml",
    )

    sum_ep_rewards = agent.evaluate()
    for episode, reward in enumerate(sum_ep_rewards):
        eval_results_df.loc[len(eval_results_df)] = {
            "corpus_size": corpus_size,
            "episode": episode + 1,
            "sum_reward": reward,
        }
    print(f"Finished evaluating corpus size {corpus_size}", flush=True)
    eval_results_df.to_csv("./results/greedy_gnn_eval_results.csv", index=False)
