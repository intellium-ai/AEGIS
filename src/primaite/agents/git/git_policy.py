import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv
from torch_geometric.nn.pool import global_max_pool
from torch.optim import Adam
from transformers import BertModel, BertTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from typing import List
import logging

LLM_PROMPT = """Your job is to defend the network against attacks. Given the provided network graph state, please choose an action integer to execute within the space. Your response should be a single integer.
Action integer: """

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
        return x


class TextEncoder(nn.Module):
    def __init__(self, model_name: str = "bert-base-uncased", device: str = 'cpu'):
        super().__init__()
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(device)

    def forward(self, graph):
        prompt = f"Node features:\n{graph.x}\n\nEdge Index: {graph.edge_index}"
        input_ids = self.tokenizer.encode(prompt)
        outputs = self.model(torch.tensor(input_ids).unsqueeze(0).to(self.device))  # type: ignore
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

class LLM(nn.Module):
    def __init__(self, model_name='HuggingFaceTB/SmolLM-1.7B-Instruct', device: str = 'cpu'):
        super().__init__()
        self.device = device
        self.model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, torch_dtype=torch.float16, padding=True)
    
    def _tokenize(self, prompt, use_template: bool = False) -> List[int]:
        ## Prepare prompt with template
        if use_template:
            messages = [{"role": "user", "content": prompt}]
            inputs=self.tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt").to(self.device)
        else:
            inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        return inputs
        
    def get_input_embeddings(self, prompt: str = None, token_ids: List[int] = None) -> torch.Tensor:
        """Your non-standard .generate"""
        assert prompt or token_ids, "A text prompt or token_ids must be passed to get_input_embeddings"
        if prompt:
            inputs = self._tokenize(prompt=prompt, use_template=False).to(self.device)
        else:
            inputs = token_ids
        with torch.no_grad():
            embs = self.model.get_input_embeddings()(inputs)
            
        return embs
    
    def generate_from_embeddings(self, text_embeddings) -> str:
        next_token_logits = self.model.forward(inputs_embeds=text_embeddings).logits[:, -1, :]
        
        # Select next token (you can use different strategies here)
        next_token = torch.argmax(next_token_logits, dim=-1)
        # Convert final embeddings back to token ids
        
        #tokens = torch.argmax(logits, dim=-1)
        #print(len(tokens[0]))
        return self.tokenizer.decode(next_token)
        
class GITPolicy(nn.Module):
    def __init__(self, action_space=None, state_space=None, hidden_dim=None, ge_learning_rate=0.0001, ap_learning_rate= 0.0001, device: str ='cpu'):
        super(GITPolicy, self).__init__()
        self.device = device

        # space size check
        assert state_space is not None, "None state_space input: state_space should be assigned."
        assert action_space is not None, "None action_space input: action_space should be assigned"
        if hidden_dim is None:
            hidden_dim = state_space * 2

        
        self.llm = LLM(device=device)
        
        # Hacky way to get the size of each tokens embedding - this helps us to align the GNN and LLM output shapes later.
        self.llm_embedding_size = self.llm.get_input_embeddings(prompt='hack').shape[2]
        
        self.ge = GraphEmbedding(in_channels=state_space, hidden_dim=hidden_dim, output_dim=self.llm_embedding_size).to(device)
        #self.ap = AlignmentProjector(in_features=self.llm_output_shape, out_features=self.llm_output_shape, out_features=self.llm_output_shape)
        #self.te = TextEncoder()
        self.ge_optimizer = Adam(self.ge.parameters(), lr=ge_learning_rate)
        
    def forward(self, x, edge_index):
        """This should return the action integer"""
        
        llm_embs= self.llm.get_input_embeddings(prompt=LLM_PROMPT)
        graph_embs = self.ge(x.to(self.device), edge_index.to(self.device))
        
         # Match graph embs output with LLM embs dimensionality and dtype.
        graph_embs = graph_embs.unsqueeze(0).to(torch.float16)
        concatenated_embs = torch.cat([graph_embs, llm_embs], 1)
        #projected_embs = self.ap(concatenated_embs)
        action_int = self.llm.generate_from_embeddings(text_embeddings=concatenated_embs)
        
        # Validate the output is an integer. If not, fall back to 0. TODO: Instead of selecting the top token and hoping it's an integer, GET the highest (integer) logit token.
        try:
            action_int = int(action_int)
        except:
            logging.error(f"The LLM did not produce an action integer! LLM output: {action_int}. Falling back to action int 0")
            action_int = 0
            
        return action_int

# def train(train_data, llm: LLM, gnn, te, criterion, opt, device):
#     gnn.train()
#     epoch_loss = 0
#     for data in train_data:
#         x, y = data.x.to(device), data.edge_index.to(device)
#         graph_output = gnn(x, y)
#         prompt = 'What action should be taken? Please specify a number. Here is the graph:\n{data.x}'.format(data.x) # dummy example
#         input_embeddings = llm.get_input_embeddings(prompt)
#         concatenated_embeddings = torch.cat([graph_output, input_embeddings], 1)
#         action_output = llm.generate_from_embeddings(concatenated_embeddings)
#         print(f"Text output: {action_output}")
        
#         # Get loss from primaite...
#         loss = criterion(text_output, graph_output, torch.ones(1).to(device))
#         epoch_loss += loss.item()
#         loss.backward()
#         opt.step()

# def main():
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f'AEGIS running on {device}')
#     dataset = GNNBenchmarkDataset(root="data/", name="PATTERN")
#     length = len(dataset)
#     train_dataset = dataset[:int(0.8*length)]
#     test_dataset = dataset[int(0.8*length):]
#     train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
#     test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)


#     # Initialise models!
#     gnn = GraphEmbedding(in_channels=dataset.num_node_features, output_dim=768, hidden_dim=512).to(device)
#     text_encoder = TextEncoder().to(device)
#     #ap = AlignmentProjector().to(device) ?????
#     llm = LLM()

#     loss = nn.CosineEmbeddingLoss()

#     optimizer = Adam(gnn.parameters(), lr=0.001)

#     for epoch in range(50):
#         print(f"====== Epoch {epoch + 1} =======")
#         #train_loss = train(train_data=train_dataloader, gnn=gnn, te=text_encoder, criterion=loss, opt=optimizer, device=device)
#         test_loss = test(test_data=test_dataloader, gnn=gnn, te=text_encoder, criterion=loss, device=device)
#         print(f"Train Loss: {train_loss}, Test Loss = {test_loss}\n\n")