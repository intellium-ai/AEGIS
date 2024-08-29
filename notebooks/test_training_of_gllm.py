from primaite.agents.aegis.gllm import GLLM
from primaite.agents.aegis.modules.openai import OpenAIClient
from primaite.agents.git_agent import GITAgent
from torch.utils.data import DataLoader
from primaite.agents.llm.utils import network_connectivity_desc
import logging

logging.disable(logging.CRITICAL)

gllm = GLLM()
openai = OpenAIClient(openai_api_key="")

questions = [
    "How many nodes are in the network?",
    "How many nodes are in the network?",
    "Given the provided network configuration, how many nodes are their in this network?"
    "How many nodes are in the network?",
    "How many nodes are in the network?",
    "How many nodes are in the network?",
    "How many nodes are in the network?",
    "How many nodes are in the network?",
    "How many nodes are in the network?",
    "How many nodes are in the network?",
    "How many nodes are in the network?",
    "How many nodes are in the network?",
    "How many nodes are in the network?",
    "How many nodes are in the network?",
    "How many nodes are in the network?",
    "How many nodes are in the network?",
    "How many nodes are in the network?",
    "How many nodes are in the network?",
    "How many nodes are in the network?",
    "How many nodes are in the network?",
    "How many nodes are in the network?",
    "How many nodes are in the network?",
    "How many nodes are in the network?",
    "How many nodes are in the network?",
    "How many nodes are in the network?",
    "How many nodes are in the network?",
    "How many nodes are in the network?",
    "How many nodes are in the network?",
    "How many nodes are in the network?",
    "How many nodes are in the network?",
    "How many nodes are in the network?",
    "How many nodes are in the network?",
    "How many nodes are in the network?",
    "How many nodes are in the network?",
    "How many nodes are in the network?",
    "How many nodes are in the network?",
    "How many nodes are in the network?",
    "How many nodes are in the network?",
    "How many nodes are in the network?",
    "How many nodes are in the network?",
    "How many nodes are in the network?",
    "How many nodes are in the network?",
    "How many nodes are in the network?",
    "How many nodes are in the network?",
    "How many nodes are in the network?",
    "How many nodes are in the network?",
    "How many nodes are in the network?",
    "How many nodes are in the network?",
    "How many nodes are in the network?",
    "How many nodes are in the network?",
    "How many nodes are in the network?",
    "How many nodes are in the network?",
    "How many nodes are in the network?",
]

# Mock a primaite graph for development
agent = GITAgent(
    training_config_path="../src/primaite/config/_package_data/training/git.yaml",
    lay_down_config_path="../src/primaite/config/_package_data/lay_down/lay_down_config_6_data_manipulation.yaml",
)
obs = agent._env.reset()
data = agent.create_graph(obs)
network_desc = network_connectivity_desc(agent._env)


import os
import pickle as pkl

if "openai_responses.pkl" not in os.listdir("./"):
    openai_responses = []
    openai_prompts = gllm.build_prompts(questions=questions, network_desc=network_desc, model="openai")
    for prompt in openai_prompts:
        openai_responses.append(openai.generate(prompt=prompt))

    with open("openai_responses.pkl", "wb") as file:
        pkl.dump(openai_responses, file)
else:
    with open("openai_responses.pkl", "rb") as file:
        openai_responses = pkl.load(file)

from primaite.agents.aegis.data import GLLMDataset, collate_fn

dataset = GLLMDataset(
    graphs=[data for _ in range(len(questions))],
    questions=[question for question in questions],
    gt_answers=[response for response in openai_responses],
    llm=gllm.llm,
)

dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

from primaite.agents.aegis.gllm import train_loop

gllm_responses = train_loop(model=gllm, dataloader=dataloader, network_desc=network_desc, n_epochs=500)
