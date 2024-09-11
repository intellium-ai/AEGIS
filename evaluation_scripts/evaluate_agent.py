from primaite.agents.gnn_agent import GNNAgent
from primaite.agents.vanilla_llm.agent import LLMAgent
from primaite.agents.sb3 import SB3Agent
from primaite.agents.git_agent import GITAgent
import matplotlib.pyplot as plt
import os
import pandas as pd

# SETUP
evaluation_agent = 'gnn' # GNN, LLM, SB3, GLLM
lay_down_folder_pth = os.listdir("../data/datasets/notebook_generated_dataset/")
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
        agent_type = GITAgent # We cannot yet save and load these :(


lay_down_folder_pth.remove(pt_dataset_name)
results = pd.DataFrame(columns=['lay_down', 'num_nodes', 'num_edges', 'av_ep_reward'])
lay_down_eval_rewards = []
for idx, lay_down in enumerate(lay_down_folder_pth):
    # Agent type
    if evaluation_agent == 'llm':
        agent = agent_type(
            training_config_path="../agents/training_configs/llm.yaml",
            lay_down_config_path=f"../data/datasets/GLLM_Graph_Dataset/{lay_down}",
            fireworks_api_key=fireworks_api_key
        )
    else:
        agent = agent_type(
            training_config_path="../agents/training_configs/llm.yaml",
            lay_down_config_path=f"../data/datasets/GLLM_Graph_Dataset/{lay_down}",
        )

    avg_ep_rewards = agent.evaluate()
    lay_down_eval_rewards.append(avg_ep_rewards)
    results.loc[len(results)] = {'lay_down': lay_down, 'num_nodes': agent._env.num_nodes, 'num_edges': agent._env.num_links, 'av_ep_reward': avg_ep_rewards}
    
# Format and save results figure
results = results.sort_values(by='num_nodes', ascending=True)
print(results)
plt.plot(results['num_nodes'].to_list(), results['av_ep_reward'].to_list())
plt.title('Laydown average episode reward')
plt.savefig('./results.png')