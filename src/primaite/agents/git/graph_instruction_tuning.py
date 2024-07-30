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
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List

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

    def forward(self, graph, device):
        prompt = f"Node features:\n{graph.x}\n\nEdge Index: {graph.edge_index}"
        input_ids = self.tokenizer.encode(prompt)
        outputs = self.model(torch.tensor(input_ids).unsqueeze(0).to(device))  # type: ignore
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

class LLM:
    def __init__(self, model='HuggingFaceTB/SmolLM-1.7B-Instruct', device='cuda:1'):
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float16).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model, torch_dtype=torch.float16, padding=True)
        
    def generate(self, prompt) -> str:
        """Your standard .generate"""
        
        ## Prepare prompt with template
        inputs = self._tokenize(prompt)
        output = self.model.generate(**inputs, max_new_tokens=69)
        return self.tokenizer.decode(output[0])
    
    def _tokenize(self, prompt) -> List[int]:
        ## Prepare prompt with template
        messages = [{"role": "user", "content": prompt}]
        inputs=self.tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt").to(self.device)
        return inputs
        
    def get_text_embeddings(self, prompt) -> torch.Tensor:
        """Your non-standard .generate"""
        inputs = self._tokenize(prompt)
        return self.model.get_input_embeddings()(inputs)
    
    def generate_from_embeddings(self, embeddings) -> str:
        
        
        
def train(train_data, gnn, te, criterion, opt, device):
    gnn.train()
    epoch_loss = 0
    for data in train_data:
        x, y = data.x.to(device), data.edge_index.to(device)
        graph_output = gnn(x, y)
        
        text_output = te(data, device=device)
        loss = criterion(text_output, graph_output, torch.ones(1).to(device))
        epoch_loss += loss.item()
        loss.backward()
        opt.step()

    return epoch_loss / len(train_data)


def test(test_data, gnn, te, criterion, device):
    gnn.eval()
    test_loss = 0

    for data in test_data:
        with torch.no_grad():
            x, y = data.x.to(device), data.edge_index.to(device)
            graph_output = gnn(x, y)
            text_output = te(data, device=device)

            loss = criterion(text_output, graph_output, torch.ones(1).to(device))
            test_loss += loss.item()

    return test_loss / len(test_data)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on {device}')
    dataset = GNNBenchmarkDataset(root="data/", name="PATTERN")
    length = len(dataset)
    train_dataset = dataset[:int(0.8*length)]
    test_dataset = dataset[int(0.8*length):]

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    gnn = GraphEmbedding(in_channels=dataset.num_node_features, output_dim=768, hidden_dim=512).to(device)
    text_encoder = TextEncoder()
    text_encoder.to(device)

    loss = nn.CosineEmbeddingLoss()

    optimizer = Adam(gnn.parameters(), lr=0.001)

    for epoch in range(50):
        print(f"====== Epoch {epoch + 1} =======")
        train_loss = train(train_data=train_dataloader, gnn=gnn, te=text_encoder, criterion=loss, opt=optimizer, device=device)
        test_loss = test(test_data=test_dataloader, gnn=gnn, te=text_encoder, criterion=loss, device=device)
        print(f"Train Loss: {train_loss}, Test Loss = {test_loss}\n\n")

def test():
    llm = LLM()
    print(llm.get_text_embeddings('Hi!'))
if __name__ == "__main__":
    test()
