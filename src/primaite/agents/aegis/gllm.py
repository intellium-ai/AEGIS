import torch
import logging
from typing import List, Tuple, Dict, Literal
import torch.nn.functional as F
from peft.tuners.lora import LoraConfig
from torch.optim import Adam
from primaite.agents.aegis.modules.llm import LLM
from primaite.agents.aegis.modules.ge import GraphEmbedding
from torch.utils.data import DataLoader
from primaite.agents.aegis.prompts import GLLM_PROMPT

logging.getLogger().setLevel(logging.INFO)


class GLLM(torch.nn.Module):
    default_peft_config = LoraConfig(task_type="CAUSAL_LM", r=12, lora_dropout=0.1)

    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolLM-1.7B-Instruct",
        ge_device: str = "cuda:0",
        ge_learning_rate: float = 0.005,
        llm_learning_rate: float = 0.005,
        peft_config: LoraConfig = default_peft_config,
    ):
        super(GLLM, self).__init__()
        self.ge_device = ge_device
        self.model_name = model_name
        self.ge_learning_rate = ge_learning_rate
        self.llm_learning_rate = llm_learning_rate

        # Initialise peft_config
        self.peft_config = peft_config

        self.llm = LLM(model_name=self.model_name, peft_config=self.peft_config)
        self.ge = GraphEmbedding(
            device=self.ge_device,
            in_channels=6,
            hidden_dim=128,
            n_tokens=100,
            n_gat_layers=1,
            output_dim=self.llm.llm_embedding_size,
        ).to(self.ge_device)

        # self.llm_optimizer = Adam(self.llm.parameters(), lr=self.llm_learning_rate)
        self.optimizer = Adam(self.parameters(), lr=self.ge_learning_rate)
        self.loss_fn = torch.nn.CosineEmbeddingLoss()

    def forward(self, graph_batch, gllm_prompts):
        """This should return the action integer"""
        # 1.0 - Get the graph token(s)
        graph_output = self.ge(graph_batch.x, graph_batch.edge_index, batch=graph_batch.batch)
        graph_embs = graph_output.to(torch.float16)  # Match graph embs with LLM dimensionality
        graph_embs = self.llm._concat_graph_tags(graph_embs)  # Add <graph>...</graph>

        # 2.0 - Get the combined embeddings of graph and text tokens.
        llm_embs = self.llm.get_input_embeddings(prompts=gllm_prompts, grad=False)

        embeddings = self.llm.format_embeddings(text_embeddings=llm_embs, graph_embeddings=graph_embs)

        # 3.0 - Generate the next action
        token_ids, probs, gllm_hidden_states = self.llm.generate_from_embeddings(
            text_embeddings=embeddings,
            max_new_tokens=20,
            restrict_output=False,
            grad=True,
            last_hidden_state=True,
        )
        print("toto bene")

        gllm_response = self.llm.tokenizer.decode(token_ids, skip_special_tokens=True)

        return gllm_response, gllm_hidden_states

    def build_prompts(
        self, questions: List[str], network_desc: str, model: Literal["openai", "gllm"] = "openai"
    ) -> List[str]:
        # Format gllm prompt. We don't format context content as this is the graph embeddings we append to the embeddings of this prompt.
        if model == "openai":
            prompts = [GLLM_PROMPT.format(question=question, context=network_desc) for question in questions]
        elif model == "gllm":
            prompts = [GLLM_PROMPT.format(question=question, context="") for question in questions]

        return prompts

    def get_cosine_embeddings_loss(self, openai_hidden_states, gllm_hidden_states):

        loss = self.loss_fn(
            openai_hidden_states.view(-1, self.llm.llm_embedding_size),
            gllm_hidden_states.view(-1, self.llm.llm_embedding_size),
            torch.ones(gllm_hidden_states.shape[0]).to("cuda:0"),
        )

        return loss


def train_loop(model, dataloader: DataLoader, network_desc: str):

    model.train(mode=True)
    # Reset openai last hidden states
    epochs = 1
    for epoch in range(epochs):
        gllm_responses = []
        gllm_hidden_states = torch.empty(0).to("cuda:0")
        model.optimizer.zero_grad()

        for batch in dataloader:
            questions = batch["questions"]
            gllm_prompts = model.build_prompts(questions=questions, network_desc=network_desc, model="gllm")

            gllm_response, last_hidden_state = model(batch["graphs"], gllm_prompts)
            gllm_responses.append(gllm_response)
            gllm_hidden_states = torch.cat([gllm_hidden_states, last_hidden_state], 0)
        loss = model.get_cosine_embeddings_loss(openai_hidden_states, gllm_hidden_states)
        print(gllm_responses[0])
        # Do the backwards pass
        loss.backward()

        # Print any None gradients
        # for name, param in model.named_parameters():
        #     print(f"Gradient for {name}: {param.grad}")

        model.optimizer.step()

        print(f"Episode {episode} Loss:", loss.item())

    return gllm_responses, openai_responses
