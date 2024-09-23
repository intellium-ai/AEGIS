from primaite.agents.gnn_agent import GNNAgent
from primaite.agents.vanilla_llm.agent import LLMAgent
from primaite.agents.sb3 import SB3Agent
from primaite.agents.git_agent import GITAgent
import matplotlib.pyplot as plt
import os
import pandas as pd

# SETUP
evaluation_agent = 'llm' # GNN, LLM, SB3, GLLM
laydown_folder_pth = '../data/datasets/notebook_generated_dataset/'
laydowns = os.listdir(laydown_folder_pth)
pt_dataset_name = 'dataset.pt'
results_df_save_pth = f'./{evaluation_agent}_results.csv'
fireworks_api_key = ''

match evaluation_agent:
    case 'gnn':
        agent_type = GNNAgent
    case 'sb3':
        agent_type = SB3Agent
    case 'llm':
        agent_type = LLMAgent
    case 'gllm':
        agent_type = GITAgent


laydowns.remove(pt_dataset_name)
results = pd.DataFrame(columns=['lay_down', 'num_nodes', 'num_edges', 'av_ep_reward'])
lay_down_eval_rewards = []
for idx, lay_down in enumerate(laydowns):
    # Agent type
    if evaluation_agent == 'llm':
        agent = agent_type(
            training_config_path=f"../agents/training_configs/{evaluation_agent}.yaml",
            lay_down_config_path=f"{laydown_folder_pth}/{lay_down}",
            fireworks_api_key=fireworks_api_key
        )
    else:
        agent = agent_type(
            training_config_path=f"../agents/training_configs/{evaluation_agent}.yaml",
            lay_down_config_path=f"{laydown_folder_pth}/{lay_down}",
        )

    avg_ep_rewards = agent.evaluate()
    lay_down_eval_rewards.append(avg_ep_rewards)
    print(agent._env.num_nodes)
    results.loc[len(results)] = {'lay_down': lay_down, 'num_nodes': agent._env.num_nodes, 'num_edges': agent._env.num_links, 'av_ep_reward': avg_ep_rewards}
    
# Format and save results figure
results = results.sort_values(by='num_nodes', ascending=True)
print(results)
results.to_csv(f'./{evaluation_agent}_evaluation_results.csv', index=False)
plt.plot(results['num_nodes'].to_list(), results['av_ep_reward'].to_list())
plt.title('Laydown average episode reward')
plt.savefig('./results.png')