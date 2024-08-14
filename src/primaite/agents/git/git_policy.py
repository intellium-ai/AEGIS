import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, LayerNorm, global_add_pool, JumpingKnowledge
from torch.optim import Adam
from transformers import BertModel, BertTokenizer, BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft.tuners.lora import LoraConfig
from typing import List, Tuple, Dict
import numpy as np
import logging


logging.getLogger().setLevel(logging.INFO)


class GraphEmbedding(nn.Module):
    def __init__(
        self, in_channels, output_dim, hidden_dim, n_tokens: int = 10, device: str = "cuda:1", n_gat_layers: int = 3
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

    def forward(self, x, edge_index):
        n_nodes = x.shape[0]
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)

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

        x = global_add_pool(x, torch.zeros(n_nodes, dtype=torch.int64).to(self.device))
        x = F.relu(x)
        x = self.linear(x)
        x = x.view(self.n_tokens, self.output_dim)  # Reshape to (n_tokens, output_dim)
        return x


class TextEncoder(nn.Module):
    def __init__(self, model_name: str = "bert-base-uncased", device: str = "cpu"):
        super().__init__()
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(device)

    def forward(self, graph):
        # This is suboptimal, but I inherited it from JackS.
        prompt = f"Node features:\n{graph.x}\n\nEdge Index: {graph.edge_index}"
        input_ids = self.tokenizer.encode(prompt)
        outputs = self.model(torch.tensor(input_ids).unsqueeze(0).to(self.device))  # type: ignore
        return outputs.last_hidden_state[:, 0, :]


class LLM(torch.nn.Module):
    def __init__(self, model_name="HuggingFaceTB/SmolLM-1.7B-Instruct", peft_config: LoraConfig = None):
        super().__init__()

        # Initialise quantisation config
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        self.peft_config = peft_config
        # Use accelerate device mapping to distribute the model across all available cuda devices.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=self.bnb_config,
            peft_config=self.peft_config,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, torch_dtype=torch.float16, padding=True, device_map="auto"
        )

        # Figure out what tokens to allow in action generate calls
        self.filtered_vocab, self.numeric_token_ids = self._get_filtered_tokenizer_vocab()

        # Get start and end graph tag embeddings
        self.graph_start_emb = self.get_embeddings(prompt="<graph>", apply_chat_tokens=False)
        self.graph_end_emb = self.get_embeddings(prompt="</graph>", apply_chat_tokens=False)
        self.close_msg_emb = self.get_embeddings(
            prompt=self.tokenizer.eos_token + "\n" + self.tokenizer.bos_token + "assistant" + "\n",
            apply_chat_tokens=False,
        )

    def _get_filtered_tokenizer_vocab(self) -> Tuple[Dict[str, int], Dict[str, int]]:

        # Allowed tokens:
        allowed_tokens = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "."]

        tokenizer_vocab = self.tokenizer.get_vocab()
        filtered_vocab = {}
        numeric_token_ids = {}
        for k, v in tokenizer_vocab.items():
            if k in allowed_tokens:
                # Allowed token
                filtered_vocab[k] = v
                if k in allowed_tokens[:-1]:
                    # numeric
                    numeric_token_ids[k] = v

        return filtered_vocab, numeric_token_ids

    def apply_chat_template(self, system: str, prompt: str, close_usr_msg: bool = False) -> str:
        bos_token = self.tokenizer.bos_token
        eos_token = self.tokenizer.eos_token
        conversation = ""
        if system:
            conversation += bos_token + "system\n" + system + eos_token + "\n"
        conversation += bos_token + "user\n" + prompt + "\n"
        if close_usr_msg:
            conversation += eos_token

        return conversation

    def _concat_graph_tags(self, graph_embs: torch.Tensor) -> torch.Tensor:
        """Given graph embeddings, concatenate the start and end tags <graph> ... </graph>"""
        return torch.cat([self.graph_start_emb, graph_embs, self.graph_end_emb], dim=1)

    def get_embeddings(
        self, prompt: str = None, token_ids: List[int] = None, system: str = None, apply_chat_tokens: bool = True
    ) -> torch.Tensor:
        """Your non-standard .generate"""
        assert prompt or token_ids, "A text prompt or list of token ids must be passed to get_embeddings"
        if prompt:
            if apply_chat_tokens:
                inputs = self.apply_chat_template(system=system, prompt=prompt, close_usr_msg=False)
            elif not apply_chat_tokens and not system:
                inputs = prompt
            inputs = self.tokenizer.encode(inputs, return_tensors="pt")
        else:
            inputs = token_ids
        with torch.no_grad():
            embs = self.model.get_input_embeddings()(inputs)

        return embs

    def generate_from_embeddings(
        self, text_embeddings: torch.Tensor, grad: bool = True, max_new_tokens: int = 10, restrict_output: bool = True
    ) -> Tuple[List[int], torch.Tensor]:
        next_token_ids = []
        next_token_probs = torch.empty(0)
        n_tokens = 0
        stop_generating = False

        # Generate stop token of max_new_tokens reached
        while not stop_generating:
            if not grad:
                with torch.no_grad():
                    next_token_embs = self.model.forward(inputs_embeds=text_embeddings)
            else:
                next_token_embs = self.model.forward(inputs_embeds=text_embeddings)

            # Get logits
            logits = next_token_embs.logits[:, -1, :]

            # Apply restriction on the output logits (or don't)
            if restrict_output:
                token_id, prob = self._filter_logits(
                    logits=logits, prev_token_id=next_token_ids[-1] if next_token_ids else None
                )
            else:
                logit = torch.max(next_token_embs.logits[:, -1, :], dim=-1)
                token_id = logit.indices[0]
                prob = logit.values

            # Check what device the output is on
            device = prob.device

            # Update probs and token ids
            next_token_ids.append(token_id)
            next_token_probs = torch.cat([next_token_probs.to(device), prob.unsqueeze(0)], dim=0)

            # Get embeddings of the new token and add it to the previous embeddings
            new_embeddings = self.get_embeddings(token_ids=torch.tensor([token_id])).unsqueeze(0)
            text_embeddings = torch.cat([text_embeddings, new_embeddings], dim=1)
            n_tokens += 1

            # Check for eos token or max_new_tokens limit reached
            if token_id == self.tokenizer.eos_token_id or n_tokens == max_new_tokens:
                stop_generating = True

        return next_token_ids, next_token_probs

    def _filter_logits(self, logits: torch.Tensor, prev_token_id: int) -> Tuple[int, torch.Tensor]:

        # Filter logits, keeping only those allowed
        candidate_tokens = {}
        for idx in range(logits.shape[-1]):
            if idx in self.filtered_vocab.values():
                candidate_tokens[idx] = logits[0][idx]

        # Sample highest prob token remaining
        token_id = max(candidate_tokens, key=lambda k: candidate_tokens[k].max().item())

        # If the token is a '.' and the previous token was not a number, resample but exclude the '.'
        if token_id == self.filtered_vocab["."] and prev_token_id not in self.numeric_token_ids.values():
            token_id = max((k for k in candidate_tokens if k != "."), key=lambda k: candidate_tokens[k].max().item())

        # If the sampled token is a '.' but the previous one was also a '.', resample ignoring the '.'
        elif prev_token_id and (token_id == self.filtered_vocab["."] and prev_token_id == self.filtered_vocab["."]):
            token_id = max((k for k in candidate_tokens if k != "."), key=lambda k: candidate_tokens[k].max().item())

        prob = candidate_tokens[token_id]
        return token_id, prob


class GITPolicy(nn.Module):
    def __init__(
        self,
        action_space: int = None,
        state_space: int = None,
        hidden_dim: int = None,
        ge_learning_rate: float = 0.0001,
        llm_device: str = "cuda:0",
        ap_device: str = "cuda:1",
        ge_device: str = "cuda:1",
        n_graph_tokens: int = 10,
    ):

        super(GITPolicy, self).__init__()
        self.llm_device = llm_device
        self.ap_device = ap_device
        self.ge_device = ge_device

        # space size check
        assert state_space is not None, "None state_space input: state_space should be assigned."
        assert action_space is not None, "None action_space input: action_space should be assigned."
        if hidden_dim is None:
            hidden_dim = state_space * 2

        self.llm = LLM(device=self.llm_device)

        # Hacky way to get the size of each tokens embedding - this helps us to align the GNN and LLM output shapes later.
        self.llm_embedding_size = self.llm.get_embeddings(prompt="hack").shape[2]

        self.ge = GraphEmbedding(
            in_channels=state_space,
            hidden_dim=hidden_dim,
            output_dim=self.llm_embedding_size,
            device=ge_device,
            n_tokens=n_graph_tokens,
        ).to(self.ge_device)

        # For tracking episode rewards and probs
        self.roll_out = []
        self.ge_optimizer = Adam(self.ge.parameters(), lr=ge_learning_rate)

    def put_data(self, data):
        self.roll_out.append(data)

    def forward(self, x, edge_index, action_prompt, reasoning_prompt):
        """This should return the action integer"""

        # 1.0 - Get the graph token(s)
        graph_output = self.ge(x, edge_index)
        graph_embs = graph_output.unsqueeze(0).to(torch.float16)  # Match graph embs with LLM dimensionality
        graph_embs = self.llm._concat_graph_tags(graph_embs)  # Add <graph>...</graph>

        # 2.0 - Reason about the network (no grad) with graph tokens too
        reasoning_embs = self.llm.get_embeddings(prompt=reasoning_prompt)
        concatenated_embs = torch.cat([reasoning_embs.to(self.llm_device), graph_embs.to(self.llm_device)], dim=1)
        concatenated_embs = torch.cat([concatenated_embs, self.llm.close_msg_emb], dim=1)

        token_ids, probs = self.llm.generate_from_embeddings(
            text_embeddings=reasoning_embs, grad=False, restrict_output=False, max_new_tokens=100
        )  # Set grad to true when more GPUage
        reasoning_statement = self.llm.tokenizer.decode(token_ids, skip_special_tokens=True)

        # 3.0 - Get the combined embeddings of graph and text tokens.
        llm_embs = self.llm.get_embeddings(prompt=action_prompt.format(reasoning_statement=reasoning_statement))
        concatenated_embs = torch.cat([llm_embs.to(self.llm_device), graph_embs.to(self.llm_device)], dim=1)
        concatenated_embs = torch.cat([concatenated_embs, self.llm.close_msg_emb], dim=1)

        # 4.0 - Generate the next action
        token_ids, probs = self.llm.generate_from_embeddings(text_embeddings=concatenated_embs, max_new_tokens=5)
        response = self.llm.tokenizer.decode(token_ids, skip_special_tokens=True)

        logging.info(f"LLM Generated Reasoning: '{reasoning_statement}' with action '{response}'")
        return response, probs, reasoning_statement

    def train_net(self, gamma: float = 0.99) -> Tuple[float, float]:
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

        # Reset gradients
        self.ge_optimizer.zero_grad()

        # Calculate loss for the episode and do backprop
        for r, prob in self.roll_out[::-1]:
            R = r + gamma * R
            loss = -prob * ((R - G_mean) / G_std)
            loss.backward()

        self.ge_optimizer.step()

        # Get average reward and reset rollout
        mean_reward = np.mean([rew[0] for rew in self.roll_out])
        self.roll_out = []

        return loss.cpu().detach().numpy(), mean_reward
