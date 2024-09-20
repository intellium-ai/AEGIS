import logging
from primaite.agents.aegis.gllm import GLLM
from torch_geometric.loader import DataLoader
import torch

logging.disable(logging.CRITICAL)

#gllm = GLLM.load(path='/srv/aegis-training-checkpoints/Sep13_15-54-epoch_4-step_5000')
gllm = GLLM()

data = torch.load('./200_items.pt')

train_dataloader = DataLoader(dataset=data.to_data_list(), batch_size=2, shuffle=True, pin_memory=True)

gllm.train_model(
    train_dataloader, 
    n_epochs=100,
    lr=1e-4,
    save_every_n_steps=10,
    save_every_n_epochs=1,
)