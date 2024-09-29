from datetime import datetime
from primaite.agents.git_agent import GITAgent
import os
import logging
import random
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

logging.disable(logging.CRITICAL)
training_laydowns = "/srv/aegis-training-data/laydowns/group_1/laydowns/"
evaluation_laydowns = "/srv/aegis-evaluation-data/eval_dataset/"
run_timestamp = datetime.now().strftime("%H-%M-%d-%m")
save_path = f"/srv/aegis-trained-agents/git_{run_timestamp}/"
corpus_sizes = ["5", "10", "20", "40"]
eval_results_df = pd.DataFrame(columns=["corpus_size", "run", "episode", "sum_reward"])

laydowns = os.listdir(training_laydowns)
random.shuffle(laydowns)

# train on 128 laydowns for 1 episode each
laydowns = laydowns[:128]

if not os.path.exists(save_path):
    os.mkdir(save_path)
first = True
for idx, laydown in enumerate(laydowns):
    if first:
        agent = GITAgent(
            training_config_path="../agents/training_configs/git.yaml",
            lay_down_config_path=training_laydowns + laydown,
            save_path="./runs/gllm_pretraining_logs_2/epoch_4-step_5000/",
        )
    else:
        agent.reset_env(laydown_config_path=training_laydowns + laydown)
        agent._agent = agent.load(save_path)

    agent.learn(episodes=1)
    agent._agent.save(save_path)
    first = False
    print(f"Finished with laydown {idx} of {len(laydowns)}")
