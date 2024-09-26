import torch
from torch import nn
from torch.optim import Adam
from typing import Tuple
import numpy as np
import logging

from peft.tuners.lora import LoraConfig

from primaite.agents.aegis.modules.llm import LLM
from primaite.agents.aegis.modules.ge import GraphEmbedding


logging.getLogger().setLevel(logging.INFO)

default_peft_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=64,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "wte"],
    lora_alpha=32,
)


class GITPolicy(nn.Module):
    def __init__(
        self,
        state_space: int = None,
        hidden_dim: int = None,
        ge_device: str = "cuda:0",
        n_graph_tokens: int = 10,
        lora_config: LoraConfig = default_peft_config,
        **kwargs,
    ):

        super(GITPolicy, self).__init__()
        self.ge_device = ge_device

        # space size check
        assert not (
            state_space is None and "ge" not in kwargs.keys()
        ), "None state_space input: state_space should be assigned."
        if hidden_dim is None and "ge" not in kwargs.keys():
            hidden_dim = state_space * 2

        if "llm" in kwargs:
            self.llm = kwargs["llm"]
        else:
            self.llm = LLM(peft_config=lora_config)

        if "ge" in kwargs:
            self.ge = kwargs["ge"]
        else:
            self.ge = GraphEmbedding(
                in_channels=state_space,
                hidden_dim=hidden_dim,
                output_dim=self.llm.llm_embedding_size,
                device=ge_device,
                n_tokens=n_graph_tokens,
            ).to(self.ge_device)

    def forward(self, graph_batch, action_prompt, reasoning_prompt):
        """This should return the action integer"""

        # 1.0 - Get the graph token(s)
        graph_output = self.ge(graph_batch)
        graph_embs = graph_output.to(torch.float16)  # Match graph embs with LLM dimensionality
        graph_embs = self.llm._concat_graph_tags(graph_embs)  # Add <graph>...</graph>

        # 2.0 - Reason about the network (no grad) with graph tokens too
        inputs = self.llm.get_input_embeddings(prompts=[reasoning_prompt])
        inputs = self.llm.format_inputs(inputs=inputs, graph_embeddings=graph_embs)

        token_ids, probs = self.llm.generate_from_embeddings(
            inputs_embeds=inputs["inputs_embeds"].to("cuda:0"),
            attention_mask=inputs["attention_mask"].to("cuda:0"),
            grad=False,
            restrict_output=False,
            max_new_tokens=40,
        )  # Set grad to true when more GPUage
        reasoning_statement = self.llm.tokenizer.decode(token_ids.squeeze(), skip_special_tokens=True)

        # 3.0 - Get the combined embeddings of graph and text tokens.
        inputs = self.llm.get_input_embeddings(prompts=[action_prompt.format(reasoning_statement=reasoning_statement)])
        inputs = self.llm.format_inputs(inputs=inputs, graph_embeddings=graph_embs)

        # 4.0 - Generate the next action (5 tokens required per action)
        token_ids, probs = self.llm.generate_from_embeddings(
            inputs_embeds=inputs["inputs_embeds"].to("cuda:0"),
            attention_mask=inputs["attention_mask"].to("cuda:0"),
            max_new_tokens=5,
        )
        response = self.llm.tokenizer.decode(token_ids.squeeze(), skip_special_tokens=True)

        logging.info(f"LLM Generated Reasoning: '{reasoning_statement}' with action '{response}'")
        return (
            response,
            probs.squeeze(),
            reasoning_statement,
            (inputs["inputs_embeds"].detach(), token_ids.squeeze().detach()),
        )

    def save(self, path: str) -> None:
        self.llm.save(path)
        self.ge.save(path)

    @classmethod
    def load(cls, path: str) -> None:
        return cls(llm=LLM.load(path), ge=GraphEmbedding.load(path))
