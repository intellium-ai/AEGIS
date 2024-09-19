from datetime import datetime
from primaite.agents.rllib import RLlibAgent
from primaite.agents.sb3 import SB3Agent
from stable_baselines3.a2c import A2C
from stable_baselines3.ppo import PPO
import os
import logging
logging.disable(logging.CRITICAL)
corpus_sizes = ['5', '10', '15', '20']
for corpus_size in corpus_sizes:
    training_laydowns = "/srv/aegis-evaluation-data/eval_dataset/corpus_size_" + corpus_size + '/'
    run_timestamp = datetime.now().strftime("%H-%M-%d-%m")
    save_path = f'/srv/aegis-trained-agents/5_sb3_{run_timestamp}/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        
    agent = SB3Agent(
        training_config_path="../agents/training_configs/sb3.yaml",
        lay_down_config_path=training_laydowns + '0.yaml',
    )
    
    agent.learn()
    current_checkpoint_path=agent._agent.save(save_path + 'model')
    print(f'Finished training of {corpus_size}, saved in {current_checkpoint_path}')
    
    