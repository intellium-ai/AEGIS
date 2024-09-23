from primaite.agents.gnn_agent import GNNAgent
from primaite.agents.sb3 import SB3Agent
import os
import pandas as pd

# SETUP
train_agent = 'gnn' # GNN, SB3
absolute_laydown_path = '/srv/aegis-datasets/laydowns_0/'
laydowns = os.listdir(absolute_laydown_path)
laydowns[0:21]
results_df_save_pth = f'./{train_agent}_results.csv'
fireworks_api_key = ''

match train_agent:
    case 'gnn':
        agent_type = GNNAgent
    case 'sb3':
        agent_type = SB3Agent

results = pd.DataFrame(columns=['lay_down', 'num_nodes', 'num_edges', 'av_ep_reward'])
lay_down_eval_rewards = []
for idx, lay_down in enumerate(laydowns):
    # Agent type
    try:
        agent = agent_type(
            training_config_path=f"../agents/training_configs/{train_agent}.yaml",
            lay_down_config_path=f"{absolute_laydown_path}/{lay_down}",
        )

        avg_ep_rewards = agent.learn()
        lay_down_eval_rewards.append(avg_ep_rewards)
        results.loc[len(results)] = {'lay_down': lay_down, 'num_nodes': agent._env.num_nodes, 'num_edges': agent._env.num_links, 'av_ep_reward': avg_ep_rewards}
    except:
        # Just incase something goes wrong, we'll know about it :)
        results.loc[len(results)] = {'lay_down': lay_down, 'num_nodes': 'FAIL', 'num_edges': 'FAIL', 'av_ep_reward': 'FAIL'}
    
    if idx % 20 == 0:
        checkpoint_path = f'./{train_agent}_checkpoint_{idx}'
        # Checkpoint the model!
        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)
        agent._agent.save(checkpoint_path)
# Format and save results
results.to_csv(f'./{train_agent}_evaluation_results.csv', index=False)