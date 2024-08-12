import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv
from torch_geometric.nn.pool import global_max_pool
from torch.optim import Adam
from transformers import BertModel, BertTokenizer, BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from typing import List, Tuple, Any, Dict
import regex as re
import logging
import numpy as np

class GraphEmbedding(nn.Module):
    def __init__(self, in_channels, output_dim, hidden_dim, device: str = 'cuda:1'):
        super().__init__()
        self.gat = GATConv(in_channels=in_channels, out_channels=hidden_dim)
        self.linear = nn.Linear(in_features=hidden_dim, out_features=output_dim)
        self.device = device
    def forward(self, x, edge_index):
        x = self.gat(x.to(self.device), edge_index.to(self.device))
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
    def __init__(self, in_features, out_features, hidden_dim, device: str = 'cuda:1') -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_features=in_features, out_features=hidden_dim)
        self.linear2 = nn.Linear(in_features=hidden_dim, out_features=out_features)
        self.device = device
        
    def forward(self, x):
        x = self.linear1(x.to(self.device))
        x = F.relu(x)
        x = self.linear2(x)
        return F.relu(x)

class LLM(torch.nn.Module):
    def __init__(self, model_name='HuggingFaceTB/SmolLM-1.7B-Instruct', device: str = 'cuda:0'):
        super().__init__()
        self.device = device
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        
        # Load model and tokenizer using bitsandbytes nf4 bit quantization and use accelerate device mapping to distribute the model across all available devices.
        self.model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map='auto', quantization_config=self.bnb_config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, torch_dtype=torch.float16, padding=True, device_map='auto')
        
        # Figure out what tokens to allow in action generate calls
        self.filtered_vocab = self._get_filtered_tokenizer_vocab()
    
    def _get_filtered_tokenizer_vocab(self) -> Dict[str, int]:
        # Filter out the tokenizers vocab to make sure we are only getting back what we would expect.
        filter = r'^[0-9]+$'
        tokenizer_vocab = self.tokenizer.get_vocab()
        filtered_vocab = {}
        for k, v in tokenizer_vocab.items():
            if re.fullmatch(filter, k):
                filtered_vocab[k] = v
            #elif k == self.tokenizer.eos_token:
            #    self.filtered_vocab[k] = self.tokenizer.eos_token_id
            elif k == self.tokenizer.bos_token:
                filtered_vocab[k] = v
            elif k == 'ass':
                filtered_vocab[k] = v
            elif k == 'istant':
                filtered_vocab[k] = v
            elif k == '\\':
                filtered_vocab[k] = v
            elif k == 'n':
                filtered_vocab[k] = v
            elif k == '.':
                filtered_vocab[k] = v
        print(filtered_vocab)
        return filtered_vocab
        
    def get_embeddings(self, prompt: str = None, system: str = None, token_ids: List[int] = None) -> torch.Tensor:
        """Your non-standard .generate"""
        assert prompt or token_ids, "A text prompt or token_ids must be passed to get_input_embeddings"
        
        if prompt:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            #inputs = self.tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt")
            inputs = self.tokenizer.encode(text=prompt, return_tensors="pt")
            #print(self.tokenizer.decode(inputs))
        else:
            inputs = token_ids
            
        with torch.no_grad():
            embs = self.model.get_input_embeddings()(inputs)
            
        return embs
        
    def generate_from_embeddings(self, text_embeddings, grad=True, max_new_tokens=10, restrict_output: bool =True) -> Tuple[List[int], torch.Tensor]:
        next_token_ids = []
        next_token_probs = torch.empty(0)
        n_tokens = 0
        stop_generating = False
        
        # Generate until otherwise
        while not stop_generating:
            if not grad:
                with torch.no_grad():
                    next_token_embs = self.model.forward(inputs_embeds=text_embeddings)
            else:
                next_token_embs = self.model.forward(inputs_embeds=text_embeddings)
            
            # Apply softmax on the last tokens logits
            logits = F.softmax(next_token_embs.logits[:, -1, :], dim=-1)
            
            # Apply the tokenizer filtering on the output logits
            token_id, prob = self._filter_logits(logits=logits, prev_token_id=next_token_ids[-1] if next_token_ids else None)
            
            # Check what device the output is on
            device = prob.device
            
            # Update logits and token ids
            next_token_ids.append(token_id)

            #print(next_token_probs).shape
            next_token_probs = torch.cat([next_token_probs.to(device), prob.unsqueeze(0)], dim=0)
            # Get embeddings of the new token and add a new dimension (to 3d like text_embeddings is)
            new_embeddings = self.get_embeddings(token_ids=torch.tensor([token_id])).unsqueeze(0)
            
            # Add the new token embeddings to the end of the previous tokens embeddings
            text_embeddings = torch.cat([text_embeddings, new_embeddings], dim=1)
            n_tokens += 1

            # Check for eos token or max_new_tokens limit reached
            if token_id == self.tokenizer.eos_token_id or n_tokens == max_new_tokens:
                 stop_generating = True
                
        return next_token_ids, next_token_probs
        
    def _filter_logits(self, logits: torch.Tensor, prev_token_id) -> Tuple[int, torch.Tensor]:
        candidate_tokens = {}
        for idx in range(logits.shape[-1]):
            if idx in self.filtered_vocab.values():
                candidate_tokens[idx] = logits[0][idx]
                print('Token:', self.tokenizer.decode([idx]))

        # TODO: Instead of taking the max token here, sample the max based on other criteria (IE: don't take another . if a . was in the previous).
        token_id = max(candidate_tokens, key=lambda k: candidate_tokens[k].max().item())
        print(f"Prev token: {prev_token_id}. Potential next token {token_id}. Dot ID {self.filtered_vocab['.']}")
        
        print(prev_token_id and token_id == self.filtered_vocab['.'] and token_id == prev_token_id)
        if prev_token_id and token_id == self.filtered_vocab['.'] and token_id == prev_token_id:
            # Disallow a second ., resample to exclude .
            print(f'no allowing {token_id} to be sampled')
            token_id = max((k for k in candidate_tokens if k != '.'), key=lambda k: candidate_tokens[k].max().item())
            print(f'Sampling {token_id} instead...')
            
        prob = candidate_tokens[token_id]
        return token_id, prob
            
class GITPolicy(nn.Module):
    def __init__(self, action_space=None, state_space=None, hidden_dim=None, ge_learning_rate=0.0001, ap_learning_rate= 0.0001, llm_device: str ='cuda:0', ap_device: str = 'cuda:1', ge_device: str = 'cuda:1'):
        
        super(GITPolicy, self).__init__()
        self.llm_device = llm_device
        self.ap_device = ap_device
        self.ge_device = ge_device

        # space size check
        assert state_space is not None, "None state_space input: state_space should be assigned."
        assert action_space is not None, "None action_space input: action_space should be assigned"
        if hidden_dim is None:
            hidden_dim = state_space * 2

        self.llm = LLM(device=self.llm_device)
        
        # Hacky way to get the size of each tokens embedding - this helps us to align the GNN and LLM output shapes later.
        self.llm_embedding_size = self.llm.get_embeddings(prompt='hack').shape[2]
        
        self.ge = GraphEmbedding(in_channels=state_space, hidden_dim=hidden_dim, output_dim=self.llm_embedding_size, device=ge_device).to(self.ge_device)
        self.ap = AlignmentProjector(in_features=self.llm_embedding_size, hidden_dim=self.llm_embedding_size, out_features=self.llm_embedding_size, device=ap_device).to(self.ap_device)
        
        self.roll_out = []
        self.ge_optimizer = Adam(self.ge.parameters(), lr=ge_learning_rate)
        self.ap_optimizer = Adam(self.ap.parameters(), lr=ap_learning_rate)

    def put_data(self, data):
        self.roll_out.append(data)
        
    def forward(self, x, edge_index, prompt):
        """This should return the action integer"""
        llm_embs= self.llm.get_embeddings(prompt=prompt) # If we append the graph tokens to this, we are gonna be adding them to the end of the chat template (<end turn> or whatever). I don't like that.
        graph_output = self.ge(x, edge_index)
        
        # Match graph embs output with LLM embs dimensionality and dtype
        graph_embs = graph_output.unsqueeze(0)
        
        # Project the graph embeddings to make them based
        graph_embs = self.ap(graph_embs).to(torch.float16)
        concatenated_embs = torch.cat([graph_embs.to(self.llm_device), llm_embs.to(self.llm_device)], 1)

        token_ids, probs = self.llm.generate_from_embeddings(text_embeddings=concatenated_embs, max_new_tokens=6)
        response = self.llm.tokenizer.decode(token_ids, skip_special_tokens=True)
        
        return response, probs

    def train_net(self, gamma) -> Tuple[float, float]:
        R = 0
        G = []
        G_t = 0

        # Whitening baseline
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
        self.ap_optimizer.step()
        mean_reward = np.mean([rew[0] for rew in self.roll_out])
        self.roll_out = []

        return loss.cpu().detach().numpy(), mean_reward