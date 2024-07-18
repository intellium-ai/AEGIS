from altair import Align, Data
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv
from torch_geometric.nn.pool import global_max_pool
from torch.optim import Adam
from torch_geometric.datasets import TUDataset, Entities, Planetoid, GNNBenchmarkDataset
from torch_geometric.loader import DataLoader
from transformers import BertModel, BertTokenizer

EMBEDDING_DIM = 64


class GraphEmbedding(nn.Module):
    def __init__(self, in_channels, output_dim, hidden_dim):
        super().__init__()
        self.gat = GATConv(in_channels=in_channels, out_channels=hidden_dim)
        self.linear = nn.Linear(in_features=hidden_dim, out_features=output_dim)

    def forward(self, x, edge_index):
        x = self.gat(x, edge_index)
        x = F.relu(x)
        x = self.linear(x)
        x = F.relu(x)
        return global_max_pool(x, batch=None)


class TextEncoder(nn.Module):
    def __init__(self, model_name: str = "bert-base-uncased"):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

    def forward(self, graph):
        prompt = f"Node features:\n{graph.x}\n\nEdge Index: {graph.edge_index}"
        input_ids = self.tokenizer.encode(prompt)
        outputs = self.model(torch.tensor(input_ids).unsqueeze(0))  # type: ignore
        return outputs.last_hidden_state[:, 0, :]


class AlignmentProjector(nn.Module):
    def __init__(self, in_features, out_features, hidden_dim) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_features=in_features, out_features=hidden_dim)
        self.linear2 = nn.Linear(in_features=hidden_dim, out_features=out_features)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return F.relu(x)


def train(train_data, gnn, te, criterion, opt):
    gnn.train()
    epoch_loss = 0
    for data in train_data:

        graph_output = gnn(data.x, data.edge_index)

        text_output = te(data)
        loss = criterion(text_output, graph_output, torch.ones(1))
        epoch_loss += loss.item()
        loss.backward()
        opt.step()

    return epoch_loss / len(train_data)


def test(test_data, gnn, te, criterion):
    gnn.eval()
    test_loss = 0

    for data in test_data:
        with torch.no_grad():
            graph_output = gnn(data.x, data.edge_index)
            text_output = te(data)

            loss = criterion(text_output, graph_output, torch.ones(1))
            test_loss += loss.item()

    return test_loss / len(test_data)


def main():
    dataset = GNNBenchmarkDataset(root="data/", name="PATTERN")
    train_dataset = dataset[:10]
    test_dataset = dataset[5:]

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    gnn = GraphEmbedding(in_channels=dataset.num_node_features, output_dim=768, hidden_dim=512)
    text_encoder = TextEncoder()

    loss = nn.CosineEmbeddingLoss()

    optimizer = Adam(gnn.parameters(), lr=0.001)

    for epoch in range(1):
        print(f"====== Epoch {epoch + 1} =======")
        train_loss = train(train_dataloader, gnn, text_encoder, loss, optimizer)
        test_loss = test(test_dataloader, gnn, text_encoder, loss)
        print(f"Train Loss: {train_loss}, Test Loss = {test_loss}\n\n")


if __name__ == "__main__":
    main()
