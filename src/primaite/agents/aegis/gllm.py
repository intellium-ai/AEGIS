import torch
import logging
from typing import List, Tuple, Dict, Literal
import torch.nn.functional as F
from peft.tuners.lora import LoraConfig
from torch.optim import Adam
from primaite.agents.aegis.modules.llm import LLM
from primaite.agents.aegis.modules.ge import GraphEmbedding
from primaite.agents.aegis.modules.openai import OpenAIClient
from primaite.agents.aegis.prompts import GLLM_PROMPT

logging.getLogger().setLevel(logging.INFO)


class GLLM(torch.nn.Module):
    default_peft_config = LoraConfig(task_type="CAUSAL_LM", r=12, lora_dropout=0.1)

    def __init__(
        self,
        openai_api_key: str,
        model_name: str = "HuggingFaceTB/SmolLM-1.7B-Instruct",
        ge_device: str = "cuda:0",
        ge_learning_rate: float = 0.05,
        llm_learning_rate: float = 0.05,
        peft_config: LoraConfig = default_peft_config,
    ):
        super().__init__()
        self.ge_device = ge_device
        self.model_name = model_name
        self.ge_learning_rate = ge_learning_rate
        self.llm_learning_rate = llm_learning_rate
        self.openai_api_key = openai_api_key

        # Initialise peft_config
        self.peft_config = peft_config

        self.llm = LLM(model_name=self.model_name, peft_config=self.peft_config)
        self.ge = GraphEmbedding(
            device=self.ge_device,
            in_channels=6,
            hidden_dim=128,
            n_tokens=20,
            n_gat_layers=3,
            output_dim=self.llm.llm_embedding_size,
        ).to(self.ge_device)
        self.openai = OpenAIClient(openai_api_key=self.openai_api_key)

        self.llm_optimizer = Adam(self.llm.parameters(), lr=self.llm_learning_rate)
        self.ge_optimizer = Adam(self.ge.parameters(), lr=self.ge_learning_rate)
        self.loss_fn = torch.nn.CosineEmbeddingLoss()

    def forward(self, x, edge_index, gllm_prompt):
        """This should return the action integer"""

        # 1.0 - Get the graph token(s)
        graph_output = self.ge(x, edge_index)
        graph_embs = graph_output.unsqueeze(0).to(torch.float16)  # Match graph embs with LLM dimensionality
        graph_embs = self.llm._concat_graph_tags(graph_embs)  # Add <graph>...</graph>

        # 2.0 - Get the combined embeddings of graph and text tokens.
        llm_embs = self.llm.get_embeddings(prompt=gllm_prompt)
        concatenated_embs = torch.cat([llm_embs, graph_embs], dim=1)
        concatenated_embs = torch.cat([concatenated_embs, self.llm.close_msg_emb], dim=1)

        # 3.0 - Generate the next action
        token_ids, probs = self.llm.generate_from_embeddings(
            text_embeddings=concatenated_embs, max_new_tokens=10, restrict_output=False, grad=False
        )
        gllm_response = self.llm.tokenizer.decode(token_ids, skip_special_tokens=True)

        return gllm_response

    def _build_prompt(self, question: str, network_desc: str, model: Literal["openai", "gllm"] = "openai") -> str:

        # Format gllm prompt. We don't format context content as this is the graph embeddings we append to the embeddings of this prompt.
        if model == "openai":
            prompt = GLLM_PROMPT.format(question=question, context=network_desc)
        elif model == "gllm":
            prompt = GLLM_PROMPT.format(question=question, context="")
        return prompt

    def train(self, x, edge_index, questions: List[str], network_desc: str):

        gllm_responses = []
        openai_responses = []

        # Get all openai responses now
        for question in questions:
            openai_prompt = self._build_prompt(question=question, network_desc=network_desc, model="openai")
            openai_responses.append(self.openai.generate(prompt=openai_prompt))

        for episode in range(50):
            for idx, question in enumerate(questions):
                gllm_prompt = self._build_prompt(question=question, network_desc=network_desc, model="gllm")
                gllm_response = self(x, edge_index, gllm_prompt)
                gllm_responses.append(gllm_response)

                # Reset gradients
                self.ge_optimizer.zero_grad()
                loss, openai_response_embs, gllm_response_embs = self.get_cosine_embeddings_loss(
                    openai_responses[idx], gllm_response
                )

                # Do the backwards pass
                loss.backward()
                self.ge_optimizer.step()
                self.llm_optimizer.step()
            print(f"Episode {episode} Loss:", loss.item())

        return gllm_responses, openai_responses

    def get_cosine_embeddings_loss(self, openai_response, gllm_response):

        openai_tokens = self.llm.tokenizer.encode(openai_response, return_tensors="pt").squeeze(0)
        gllm_tokens = self.llm.tokenizer.encode(gllm_response, return_tensors="pt").squeeze(0)

        openai_tokens, gllm_tokens = self._pad_smallest_text(openai_tokens=openai_tokens, gllm_tokens=gllm_tokens)

        openai_response_embs = self.llm.get_embeddings(token_ids=openai_tokens, apply_chat_tokens=False, grad=True)
        gllm_response_embs = self.llm.get_embeddings(token_ids=gllm_tokens, apply_chat_tokens=False, grad=True)
        # try:
        loss = self.loss_fn(
            gllm_response_embs, openai_response_embs, torch.ones(openai_response_embs.shape[0]).to("cuda:0")
        )
        # except:
        #     loss = -1

        return loss, openai_response_embs, gllm_response_embs

    def _pad_smallest_text(self, openai_tokens, gllm_tokens):
        if len(openai_tokens) > len(gllm_tokens):
            n_pad = len(openai_tokens) - len(gllm_tokens)
            gllm_tokens = torch.cat(
                [gllm_tokens, torch.tensor([self.llm.tokenizer.pad_token_id for _ in range(n_pad)])]
            )
        elif len(gllm_tokens) > len(openai_tokens):
            n_pad = len(gllm_tokens) - len(openai_tokens)
            openai_tokens = torch.cat(
                [openai_tokens, torch.tensor([self.llm.tokenizer.pad_token_id for _ in range(n_pad)])]
            )
        gllm_tokens = gllm_tokens.squeeze(0)
        openai_tokens = openai_tokens.squeeze(0)

        return openai_tokens, gllm_tokens
