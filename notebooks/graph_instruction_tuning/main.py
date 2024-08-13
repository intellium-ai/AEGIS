import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv
from torch_geometric.nn.pool import global_mean_pool
from transformers import BertModel, BertTokenizer


class GraphEmbedding(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphEmbedding, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.conv3 = GATConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


class TextEncoder(nn.Module):
    def __init__(self, model_name: str = "bert-base-uncased"):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

    def forward(self, x, edge_index, batch):
        prompts = f"Node features:\n{x}\n\n"

        encoded = self.tokenizer.encode(prompts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(
            "cuda:0"
        )

        outputs = self.model(encoded)
        return outputs.last_hidden_state[:, 0, :]


def test(test_data, gnn, te, criterion):
    gnn.eval()
    test_loss = 0

    for data in test_data:
        with torch.no_grad():
            graph_output = gnn(data.x, data.edge_index, data.batch)
            text_output = te(data.x, data.edge_index, data.batch)

            loss = criterion(text_output, graph_output, torch.ones(1).to("cuda:0"))
            test_loss += loss.item()

    return test_loss / len(test_data)


def train(train_data, gnn, te, criterion, opt):
    gnn.train()
    epoch_loss = 0
    for data in train_data:

        graph_output = gnn(data.x, data.edge_index, data.batch)
        text_output = te(data.x, data.edge_index, data.batch)

        loss = criterion(text_output, graph_output, torch.ones(1).to("cuda:0"))
        epoch_loss += loss.item()
        loss.backward()
        opt.step()

    return epoch_loss / len(train_data)


def main():
    dataset = GNNBenchmarkDataset(root="../../data/", name="CIFAR10").to("cuda:0")

    train_test_split = 0.75
    split_idx = int(train_test_split * len(dataset))

    train_dataset = dataset[:split_idx]
    test_dataset = dataset[split_idx:]

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    gnn = GraphEmbedding(in_channels=dataset.num_features, out_channels=768, hidden_channels=256).to("cuda:0")
    llm = TextEncoder().to("cuda:0")

    criterion = nn.CosineEmbeddingLoss()
    optimizer = Adam(gnn.parameters(), lr=0.001, weight_decay=1e-5)

    train_losses = []
    test_losses = []

    for epoch in range(10):

        best_test_loss = 10000000

        train_loss = train(train_dataloader, gnn, llm, criterion, optimizer)
        train_losses.append(train_loss)
        test_loss = test(test_dataloader, gnn, llm, criterion)
        test_losses.append(test_loss)

        print(f"====== Epoch {epoch + 1} =======")
        print(f"Train Loss: {train_loss}, Test Loss = {test_loss}")

        if test_loss < best_test_loss:
            torch.save(gnn.state_dict(), f"model_epoch_{epoch}.pt")
            print("Best loss so far, saving model")

        print("\n")

    torch.save(gnn.state_dict(), "model_final.pt")
    plt.plot(range(len(train_losses)), train_losses, label="Train")
    plt.plot(range(len(test_losses)), test_losses, label="Test")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("learning_curve.pdf", format="pdf")


if __name__ == "__main__":
    main()
