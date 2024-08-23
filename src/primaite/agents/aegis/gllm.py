import torch
import logging
from typing import List, Literal
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

        self.optimizer = Adam(self.parameters(), lr=self.ge_learning_rate)
        self.loss_fn = torch.nn.CosineEmbeddingLoss()

    def forward(self, graph_batch, gllm_prompts):
        """This should return the action integer"""
        # 1.0 - Get the graph token(s)
        graph_embs = self.ge(graph_batch).to(torch.float16)
        graph_embs = self.llm._concat_graph_tags(graph_embs)  # Add <graph>...</graph>

        # 2.0 - Get the combined embeddings of graph and text tokens.
        inputs = self.llm.get_input_embeddings(prompts=gllm_prompts, grad=False)

        # 3.0 - Add BOS, graph token embs, text token embs and EOS token - shifting padding along to the right as required to maintain standardization of sequence length.
        inputs = self.llm.format_inputs(inputs=inputs, graph_embeddings=graph_embs)

        # 3.0 - Generate the response
        token_ids, probs, gllm_hidden_states = self.llm.generate_from_embeddings(
            inputs_embeds=inputs["inputs_embeds"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=10,
            restrict_output=False,
            grad=True,
            last_hidden_state=True,
        )

        gllm_responses = self.llm.tokenizer.batch_decode(token_ids, skip_special_tokens=True)

        return gllm_responses, gllm_hidden_states

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
            openai_hidden_states.view(-1, self.llm.llm_embedding_size).to(gllm_hidden_states.device),
            gllm_hidden_states.view(-1, self.llm.llm_embedding_size),
            torch.ones(gllm_hidden_states.shape[0]).to(gllm_hidden_states.device),
        )

        return loss


def train_loop(model: GLLM, dataloader: DataLoader, network_desc: str, n_epochs: int = 10):
    model.train(mode=True)

    for epoch in range(n_epochs):
        gllm_responses = []

        for batch in dataloader:
            model.optimizer.zero_grad()
            questions = batch["questions"]
            gllm_prompts = model.build_prompts(questions=questions, network_desc=network_desc, model="gllm")

            gllm_texts, gllm_hidden_states = model(batch["graphs"], gllm_prompts)
            gllm_responses.extend(gllm_texts)

            loss = model.get_cosine_embeddings_loss(batch["gt_hidden_states"], gllm_hidden_states)
            # Do the backwards pass
            loss.backward()

            # Clear up any gpu memory that may be holding onto tensors unnecessarily
            torch.cuda.empty_cache()

        model.optimizer.step()

        print(f"Epoch {epoch} Loss:", loss.item(), gllm_responses[0].replace("\n", "//n"))

    return gllm_responses
