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
        ge_learning_rate: float = 10000.0,
        llm_learning_rate: float = 10000.0,
        peft_config: LoraConfig = default_peft_config,
    ):
        super(GLLM, self).__init__()
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

        #self.llm_optimizer = Adam(self.llm.parameters(), lr=self.llm_learning_rate)
        self.optimizer = Adam(self.parameters(), lr=self.ge_learning_rate)
        self.loss_fn = torch.nn.CosineEmbeddingLoss()

    def forward(self, x, edge_index, gllm_prompt):
        """This should return the action integer"""
        # 1.0 - Get the graph token(s)
        graph_output = self.ge(x, edge_index)
        graph_embs = graph_output.unsqueeze(0).to(torch.float16)  # Match graph embs with LLM dimensionality
        graph_embs = self.llm._concat_graph_tags(graph_embs)  # Add <graph>...</graph>
        
        # 2.0 - Get the combined embeddings of graph and text tokens.
        llm_embs = self.llm.get_embeddings(prompt=gllm_prompt, grad=False)
        concatenated_embs = torch.cat([llm_embs, graph_embs], dim=1)
        concatenated_embs = torch.cat([concatenated_embs, self.llm.close_msg_emb], dim=1)

        # 3.0 - Generate the next action
        token_ids, probs = self.llm.generate_from_embeddings(
            text_embeddings=concatenated_embs, max_new_tokens=10, restrict_output=False, grad=True
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
    
    def get_cosine_embeddings_loss(self, openai_response, gllm_response):

        openai_tokens = self.llm.tokenizer.batch_encode_plus(openai_response, return_tensors="pt", padding=True).input_ids
        gllm_tokens = self.llm.tokenizer.batch_encode_plus(gllm_response, return_tensors="pt", padding=True).input_ids
        
        openai_tokens, gllm_tokens = self._pad_smallest_texts(openai_tokens=openai_tokens, gllm_tokens=gllm_tokens)

        openai_response_embs = self.llm.get_embeddings(token_ids=openai_tokens, apply_chat_tokens=False, grad=True).view(-1, self.llm.llm_embedding_size)
        gllm_response_embs = self.llm.get_embeddings(token_ids=gllm_tokens, apply_chat_tokens=False, grad=True).view(-1, self.llm.llm_embedding_size)
        # try:
        loss = self.loss_fn(
            gllm_response_embs, openai_response_embs, torch.ones(openai_response_embs.shape[0]).to("cuda:0")
        )
        # except:
        #     loss = -1

        return loss

    def _pad_smallest_texts(self, openai_tokens, gllm_tokens):
        if openai_tokens.shape[-1] > gllm_tokens.shape[-1]:
            n_pad = openai_tokens.shape[-1] - gllm_tokens.shape[-1]
            gllm_tokens = F.pad(gllm_tokens, (0, n_pad), "constant", self.llm.tokenizer.pad_token_id)
        elif gllm_tokens.shape[-1] > openai_tokens.shape[-1]:
            n_pad = gllm_tokens.shape[-1] - openai_tokens.shape[-1]
            openai_tokens = F.pad(openai_tokens, (0, n_pad), "constant", self.llm.tokenizer.pad_token_id)

        return openai_tokens, gllm_tokens


def train_loop(model, x, edge_index, questions: List[str], network_desc: str):

    gllm_responses = []
    openai_responses = []
    model.train(mode=True)
    # Get all openai responses now
    for question in questions:
        openai_prompt = model._build_prompt(question=question, network_desc=network_desc, model="openai")
        openai_responses.append(model.openai.generate(prompt=openai_prompt))

    for episode in range(50):
        # Reset
        gllm_responses = []
        model.optimizer.zero_grad()
        for idx, question in enumerate(questions):
            gllm_prompt = model._build_prompt(question=question, network_desc=network_desc, model="gllm")
            gllm_response = model(x, edge_index, gllm_prompt)
            gllm_responses.append(gllm_response)
            
        loss = model.get_cosine_embeddings_loss(
            openai_responses, gllm_responses
        )

        # Do the backwards pass
        loss.backward()
        
        # Print gradients
        for name, param in model.named_parameters():
            print(f"Gradient of {name}: {param.grad}")
        
        model.optimizer.step()
        print(f"Episode {episode} Loss:", loss.item())

    return gllm_responses, openai_responses