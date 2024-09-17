import logging

logging.disable(logging.CRITICAL)

from primaite.agents.aegis.gllm import GLLM

gllm = GLLM.load(path='./runs/Sep13_15-54-54_ds-turing02/epoch_4-step_5000/')
from torch_geometric.loader import DataLoader
import torch

data = torch.load('/srv/aegis-training-data/laydowns/group_2/data_0.pt')

train_dataloader = DataLoader(dataset=data.to_data_list(), batch_size=2, shuffle=True, pin_memory=True)

gllm.train_model(
    train_dataloader, 
    n_epochs=5, 
    lr=0.005,
    gradient_accumulation_steps=6,
    save_every_n_steps=50,
)