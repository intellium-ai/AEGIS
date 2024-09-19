from datetime import datetime
from primaite.agents.gnn_agent import GNNAgent
import os
import logging
import random
logging.disable(logging.CRITICAL)
training_laydowns = "/srv/aegis-training-data/laydowns/group_1/laydowns/"
run_timestamp = datetime.now().strftime("%H-%M-%d-%m")
save_path = f'/srv/aegis-trained-agents/gnn_{run_timestamp}/'
if not os.path.exists(save_path):
    os.mkdir(save_path)
first = True
laydowns = os.listdir(training_laydowns)
random.shuffle(laydowns)
for idx, laydown in enumerate(laydowns):
    agent = GNNAgent(
        training_config_path="../agents/training_configs/gnn.yaml",
        lay_down_config_path=training_laydowns + laydown,
    )
    if not first:
        agent._agent = agent._agent.load(save_path)
    try:
        agent.learn()
    except Exception as e:
        print(f'Something went wrong with laydown {laydown}:\n{e}')
        pass
    agent._agent.save(save_path)
    first = False
    print(f'Finished with laydown {idx} of {len(os.listdir(training_laydowns))}')