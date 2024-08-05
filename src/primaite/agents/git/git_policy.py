import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv
from torch_geometric.nn.pool import global_max_pool
from torch.optim import Adam
from transformers import BertModel, BertTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from typing import List, Tuple
import logging
import numpy as np
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
    
    def generate_from_embeddings(self, text_embeddings) -> Tuple[str, float]:
        #with torch.no_grad():
        next_token_logit = self.model.forward(inputs_embeds=text_embeddings).logits[:, -1, :]
        
        # Select next token and prob(you can use different strategies here)
        logit = torch.max(next_token_logit, dim=-1)
    
    
        return logit
        
class GITPolicy(nn.Module):
    def __init__(self, action_space=None, state_space=None, hidden_dim=None, ge_learning_rate=0.0001, ap_learning_rate= 0.0001, device: str ='cpu'):
        super(GITPolicy, self).__init__()
        self.device = device

        # space size check
        assert state_space is not None, "None state_space input: state_space should be assigned."
        assert action_space is not None, "None action_space input: action_space should be assigned"
        if hidden_dim is None:
            hidden_dim = state_space * 2

        self.llm = LLM(device='cuda:1')
        
        # Hacky way to get the size of each tokens embedding - this helps us to align the GNN and LLM output shapes later.
        self.llm_embedding_size = self.llm.get_input_embeddings(prompt='hack').shape[2]
        
        self.ge = GraphEmbedding(in_channels=state_space, hidden_dim=hidden_dim, output_dim=self.llm_embedding_size).to(device)
        #self.ap = AlignmentProjector(in_features=self.llm_output_shape, out_features=self.llm_output_shape, out_features=self.llm_output_shape)
        #self.te = TextEncoder()
        
        self.roll_out = []
        self.ge_optimizer = Adam(self.ge.parameters(), lr=ge_learning_rate)

    def put_data(self, data):
        self.roll_out.append(data)
        
    def forward(self, x, edge_index):
        """This should return the action integer"""
        prompt = LLM_PROMPT
        llm_embs= self.llm.get_input_embeddings(prompt=prompt)
        graph_output = self.ge(x.to(self.device), edge_index.to(self.device))
        
         # Match graph embs output with LLM embs dimensionality and dtype.
        graph_embs = graph_output.unsqueeze(0).to(torch.float16)
        concatenated_embs = torch.cat([graph_embs.to('cuda:1'), llm_embs], 1)
        #projected_embs = self.ap(concatenated_embs)
        token_prob = self.llm.generate_from_embeddings(text_embeddings=concatenated_embs)
        
        # Hack to use the gradient fn from the GNN output rather than the LLM.
        #token_prob.values.grad_fn = graph_output.grad_fn
        return token_prob

    def train_net(self, gamma) -> float:
        R = 0
        G = []
        G_t = 0

        # Whitening baseline - reverse roll out and 
        for r, prob in self.roll_out[::-1]:
            G_t = r + gamma * G_t
            G.append(G_t)

        G = np.array(G)
        G_mean = G.mean()
        G_std = G.std()

        self.ge_optimizer.zero_grad()

        for r, prob in self.roll_out[::-1]:
            R = r + gamma * R
            loss = -prob * ((R - G_mean) / G_std)
            loss.backward()
        self.ge_optimizer.step()
        self.roll_out = []
        
        return loss