import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv
from torch_geometric.nn.pool import global_max_pool
from torch.optim import Adam
from transformers import BertModel, BertTokenizer, BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from typing import List, Tuple, Any
import logging
import numpy as np
LLM_PROMPT = """Your job is to defend the network against attacks. Given the provided network graph tokens, please choose one action to execute within the environment. Baring in mind that you will be rewarded for taking the most suitible action in a timely manner and with consideration for what nodes might take the highest priority.

The nodes and their respective node IDin the network are:
{node_ids}

The services and their respective service ID in the network are:
{services}

Nodes and the services (Service ID) they have running are shown below:
{node_services}

The actions you could take are laid out below:
1: TURN_ON - Turn on a node
2: TURN_OFF - Turn off a node
3: RESET - Reset a node
4: PATCH_HARDWARE - Patch a nodes hardware
5: PATCH_SERVICE - Patch a nodes service

For action 5, you must always specify the service ID to patch for example 2.

You must always state which node number this action is to be applied to. If the action is a service patch, always specify which service id to patch.

Here are some examples of actions in the format NODE_ID ACTION_ID
Action: 'RESET 1'
Action: 'PATCH_HARDWARE 2'
Action: 'NONE'
Action: 'PATCH SERVICE TCP 7'
Action: 'TURN_OFF 3'
Action: 'PATCH SERVICE UDP 5'

Action: 1.1 - Turns on CLIENT_1
Action: 2.3 - Resets CLIENT_2
Action: 5.5.1 - Patches the TCP service for"""

Your actions should always use this same format. If no action is required, just say 'NONE'.

You need to be aware of recent changes in the networks state, here is a breakdown of what has been happening:
{obs_act_history}

Now, the following changes have occurred:
{current_obs_diff}

Specify an action to take as shown above. Your turn!"""


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
        
        
    def get_embeddings(self, prompt: str = None, system: str = None, token_ids: List[int] = None) -> torch.Tensor:
        """Your non-standard .generate"""
        assert prompt or token_ids, "A text prompt or token_ids must be passed to get_input_embeddings"
        
        if prompt:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            print(messages)
            inputs = self.tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt")
        else:
            inputs = token_ids
            
        with torch.no_grad():
            embs = self.model.get_input_embeddings()(inputs)
            
        return embs
        
    def generate_from_embeddings(self, text_embeddings, grad=True, max_new_tokens=10) -> Tuple[List[int], torch.Tensor]:
        next_token_ids = []
        next_token_probs = torch.tensor(())
        n_tokens = 0
        stop_generating = False
        
        # Generate until otherwise
        while not stop_generating:
            if not grad:
                with torch.no_grad():
                    next_token_embs = self.model.forward(inputs_embeds=text_embeddings)
            else:
                next_token_embs = self.model.forward(inputs_embeds=text_embeddings)
            logit =  torch.max(next_token_embs.logits[:, -1, :], dim=-1)
            
            # Update the logits
            device = logit.indices.device
            next_token_ids.append(logit.indices[0])
            next_token_probs = torch.cat([next_token_probs.to(device), logit.values], dim=0)
            # Get embeddings of the new token and add a new dimension (to 3d like text_embeddings is)
            new_embeddings = self.get_embeddings(token_ids=logit.indices).unsqueeze(0)
            
            # Add the new token embeddings to the end of the previous tokens embeddings
            text_embeddings = torch.cat([text_embeddings, new_embeddings], dim=1)
            n_tokens += 1

            # Check for eos token or max_new_tokens limit reached
            if logit.indices == self.tokenizer.eos_token_id or n_tokens == max_new_tokens:
                stop_generating = True
                
        return next_token_ids, next_token_probs
        
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
        
    def forward(self, x, edge_index):
        """This should return the action integer"""
        prompt = LLM_PROMPT
        llm_embs= self.llm.get_embeddings(prompt=prompt)
        graph_output = self.ge(x, edge_index)
        
        # Match graph embs output with LLM embs dimensionality and dtype
        graph_embs = graph_output.unsqueeze(0)
        
        # Project the graph embeddings to make them based
        graph_embs = self.ap(graph_embs).to(torch.float16)
        concatenated_embs = torch.cat([graph_embs.to(self.llm_device), llm_embs.to(self.llm_device)], 1)

        token_ids, probs = self.llm.generate_from_embeddings(text_embeddings=concatenated_embs, max_new_tokens=2)
        
        return token_ids, probs#, graph_output

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