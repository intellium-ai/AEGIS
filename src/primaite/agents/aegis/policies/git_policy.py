import torch
from torch import nn
from torch.optim import Adam
from typing import Tuple
import numpy as np
import logging

from primaite.agents.aegis.modules.llm import LLM
from primaite.agents.aegis.modules.ge import GraphEmbedding


logging.getLogger().setLevel(logging.INFO)


class GITPolicy(nn.Module):
    def __init__(
        self,
        state_space: int = None,
        hidden_dim: int = None,
        ge_learning_rate: float = 0.0001,
        llm_device: str = "cuda:0",
        ge_device: str = "cuda:1",
        n_graph_tokens: int = 10,
    ):

        super(GITPolicy, self).__init__()
        self.llm_device = llm_device
        self.ge_device = ge_device

        # space size check
        assert state_space is not None, "None state_space input: state_space should be assigned."
        if hidden_dim is None:
            hidden_dim = state_space * 2

        self.llm = LLM()

        self.ge = GraphEmbedding(
            in_channels=state_space,
            hidden_dim=hidden_dim,
            output_dim=self.llm.llm_embedding_size,
            device=ge_device,
            n_tokens=n_graph_tokens,
        ).to(self.ge_device)

        # For tracking episode rewards and probs
        self.roll_out = []
        self.optimizer = Adam(self.parameters(), lr=ge_learning_rate)

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
            text_embeddings=reasoning_embs, grad=False, restrict_output=False, max_new_tokens=1
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
        G_std = G.std() + 1e-8 # add small episilon to avoid div by 0

        # Reset gradients
        self.optimizer.zero_grad()

        # Calculate loss for the episode and do backprop
        total_loss = 0
        for r, prob in self.roll_out[::-1]:
            R = r + gamma * R
            for p in prob:
                loss = -p * ((R - G_mean) / G_std)
                total_loss += loss
            
        total_loss.backward()
        print('Total loss:', total_loss)
        for name, param in self.named_parameters():
            print(f"Gradient of {name}: {param.grad}")

        # What are the gradients saying?
        # Print gradients
        # print('Total loss:', total_loss)
        # for name, param in self.named_parameters():
            # print(f"Gradient of {name}: {param.grad}")
        
        self.optimizer.step()

        # Get average reward and reset rollout
        mean_reward = np.mean([rew[0] for rew in self.roll_out])
        self.roll_out = []

        return total_loss.cpu().detach().numpy(), mean_reward
