from datetime import datetime
from primaite.agents.gnn_agent import GNNAgent
import os
import logging
import random
import pandas as pd
logging.disable(logging.CRITICAL)
training_laydowns = "/srv/aegis-training-data/laydowns/group_1/laydowns/"
evaluation_laydowns = '/srv/aegis-evaluation-data/eval_dataset/'
run_timestamp = datetime.now().strftime("%H-%M-%d-%m")
save_path = f'/srv/aegis-trained-agents/gnn_{run_timestamp}/'
corpus_sizes = ["5", "10", "20", "40"]
eval_results_df = pd.DataFrame(columns=["corpus_size", "run", "episode", "sum_reward"])

laydowns = os.listdir(training_laydowns)
random.shuffle(laydowns)

# train on 50 laydowns
laydowns = laydowns[:50]

if not os.path.exists(save_path):
    os.mkdir(save_path)
first = True
for idx, laydown in enumerate(laydowns):

    agent = GNNAgent(
        training_config_path="../agents/training_configs/gnn.yaml",
        lay_down_config_path=training_laydowns + laydown,
    )
    if not first:
        agent.actor = agent.actor.load(save_path)
        agent.critic = agent.critic.load(save_path)
    agent.learn()
    agent.actor.save(save_path)
    agent.critic.save(save_path)
    first = False
    print(f'Finished with laydown {idx} of {len(laydowns)}')
    
# Now evaluate the model!
for corpus_size in corpus_sizes:
    agent = GNNAgent(
        training_config_path="../agents/training_configs/gnn.yaml",
        lay_down_config_path=evaluation_laydowns + corpus_size + '_nodes.yaml',
    )
    
    agent.actor = agent.actor.load(save_path)
    agent.critic = agent.critic.load(save_path)
    
    sum_ep_rewards = agent.evaluate()
    for episode, reward in enumerate(sum_ep_rewards):
        eval_results_df.loc[len(eval_results_df)] = {
            "corpus_size": corpus_size,
            "episode": episode + 1,
            "sum_reward": reward,
        }

    eval_results_df.to_csv("gnn_eval_results.csv", index=False)