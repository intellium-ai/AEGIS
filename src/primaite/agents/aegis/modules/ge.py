import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, LayerNorm, global_add_pool, JumpingKnowledge
from torch_geometric.data.batch import Batch
import logging
import json
import os


logging.getLogger().setLevel(logging.INFO)


class GraphEmbedding(nn.Module):
    def __init__(
        self, in_channels, output_dim, hidden_dim, n_tokens: int = 100, device: str = "cuda:0", n_gat_layers: int = 3
    ):
        super().__init__()
        self.gat = GATConv(in_channels=in_channels, out_channels=hidden_dim)
        self.inner_gat = GATConv(in_channels=hidden_dim, out_channels=hidden_dim)
        self.layer_norm = LayerNorm(hidden_dim)
        self.jumping_knowledge = JumpingKnowledge(mode="max")
        self.linear = nn.Linear(in_features=hidden_dim, out_features=n_tokens * output_dim)
        self.device = device
        self.n_tokens = n_tokens
        self.output_dim = output_dim
        self.n_gat_layers = n_gat_layers

    def forward(self, graph_batch):
        x = graph_batch.x.to(self.device)
        edge_index = graph_batch.edge_index.to(self.device)

        x = self.gat(x, edge_index)
        if self.n_gat_layers > 1:
            xs = []
            for _ in range(self.n_gat_layers):
                x = self.inner_gat(x, edge_index)
                x = self.layer_norm(x)
                xs.append(x)
            x = self.jumping_knowledge(xs)
        else:
            x = x

        x = global_add_pool(x=x, batch=graph_batch.batch.to(self.device))
        x = F.relu(x)
        x = self.linear(x)
        x = x.view(graph_batch.batch_size, self.n_tokens, self.output_dim)  # Reshape to (n_tokens, output_dim)
        return x
    
    def save(self, path: str) -> None:
        # Save init arguments
        init_kwargs = {
            'in_channels' : self.gat.in_channels,
            'output_dim' : self.output_dim,
            'hidden_dim' : self.gat.out_channels,
            'n_tokens' : self.n_tokens,
            'device' : self.device,
            'n_gat_layers' : self.n_gat_layers
        }

        json.dump(init_kwargs, open(os.path.join(path, 'ge_init_kwargs.json'), 'w'))

        # Save model
        torch.save(self.state_dict(), os.path.join(path, 'ge.pt'))

    @classmethod
    def load(cls, path: str):
        init_kwargs = json.load(open(os.path.join(path, 'ge_init_kwargs.json')))
        ge = cls(**init_kwargs)
        ge.load_state_dict(torch.load(os.path.join(path, 'ge.pt')))
        return ge.to(ge.device)