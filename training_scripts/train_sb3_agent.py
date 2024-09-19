from datetime import datetime
from primaite.agents.rllib import RLlibAgent
from primaite.agents.sb3 import SB3Agent
from stable_baselines3.a2c import A2C
import os
import logging
logging.disable(logging.CRITICAL)
training_laydowns = "/srv/aegis-training-data/laydowns/group_1/laydowns/"
run_timestamp = datetime.now().strftime("%H-%M-%d-%m")
save_path = f'/srv/aegis-trained-agents/sb3{run_timestamp}/'
if not os.path.exists(save_path):
    os.mkdir(save_path)
first = True
for idx, laydown in enumerate(os.listdir(training_laydowns)):
    agent = SB3Agent(
        training_config_path="../agents/training_configs/sb3.yaml",
        lay_down_config_path=training_laydowns + laydown,
        
    )
    if not first:
        agent._agent = A2C.load('./models/model.zip', env=agent._env)
    
    agent.learn()
    print('Did a round!!')
    current_checkpoint_path=agent._agent.save(save_path)
    first = False
    print(f'Finished with laydown {idx} of {len(os.listdir(training_laydowns))}')