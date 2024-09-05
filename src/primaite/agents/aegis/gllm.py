import torch
import logging
from typing import List, Literal, Tuple
from peft.tuners.lora import LoraConfig
from torch.optim import Adam
from primaite.agents.aegis.modules.llm import LLM
from primaite.agents.aegis.modules.ge import GraphEmbedding
from torch.utils.data import DataLoader
from primaite.agents.aegis.prompts import GLLM_PROMPT
import matplotlib.pyplot as plt
import os
from datetime import datetime

logging.getLogger().setLevel(logging.INFO)


class GLLM(torch.nn.Module):
    default_peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=64,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "wte"],
        lora_alpha=32,
    )

    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolLM-1.7B-Instruct",
        ge_device: str = "cuda:0",
        learning_rate: float = 0.005,
        peft_config: LoraConfig = default_peft_config,
    ):
        super(GLLM, self).__init__()
        self.ge_device = ge_device
        self.model_name = model_name
        self.learning_rate = learning_rate

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

        self.optimizer = Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, graph_batch, gllm_prompts) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """This should return the action integer"""
        # 1.0 - Get the graph token(s)
        graph_embs = self.ge(graph_batch).to(torch.float16)
        graph_embs = self.llm._concat_graph_tags(graph_embs)  # Add <graph>...</graph>

        # 2.0 - Get the combined embeddings of graph and text tokens.
        inputs = self.llm.get_input_embeddings(prompts=gllm_prompts, grad=False)

        # 3.0 - Add BOS, graph token embs, text token embs and EOS token - shifting padding along to the right as required to maintain standardization of sequence length.
        inputs = self.llm.format_inputs(inputs=inputs, graph_embeddings=graph_embs)

        # 3.0 - Generate the response
        token_ids, gllm_probs, gllm_hidden_states = self.llm.generate_from_embeddings(
            inputs_embeds=inputs["inputs_embeds"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=100,
            restrict_output=False,
            grad=True,
            last_hidden_state=True,
        )

        return token_ids, gllm_probs, gllm_hidden_states

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
        loss_fn = torch.nn.CosineEmbeddingLoss()
        loss = loss_fn(
            openai_hidden_states.view(-1, self.llm.llm_embedding_size).to(gllm_hidden_states.device),
            gllm_hidden_states.view(-1, self.llm.llm_embedding_size),
            torch.ones(gllm_hidden_states.shape[0]).to(gllm_hidden_states.device),
        )

        return loss

    def get_crossentropy_loss(self, gt_probs, gt_tokens, gllm_probs, gllm_tokens):
        loss_fn = torch.nn.CrossEntropyLoss()
        padding_logit = torch.zeros([len(self.llm.tokenizer)])
        padding_logit[self.llm.tokenizer.pad_token_id] = (
            100  # Mock pad token of gt to 100% prob (it gets ignored anyway because of the mask)
        )
        padding_token = torch.tensor([self.llm.tokenizer.pad_token_id])  # Set padding token tensor

        # Pad the shortest set of tensors if required
        batch_size = gllm_probs.shape[0]
        if gt_probs.shape[1] > gllm_probs.shape[1]:
            # gt sequences are longer - truncate the gt responses
            gt_probs = gt_probs[:, : gllm_probs.shape[1], :]
            gt_tokens = gt_tokens[:, : gllm_tokens.shape[1]]

        elif gllm_probs.shape[1] > gt_probs.shape[1]:
            # gllm sequences are longer - pad openai responses
            n_pads = gllm_probs.shape[1] - gt_probs.shape[1]
            padding_logits = padding_logit.repeat(batch_size, n_pads, 1)

            gt_probs = torch.cat([gt_probs, padding_logits], dim=1)

            padding_tokens = padding_token.repeat(batch_size, n_pads)
            gt_tokens = torch.cat([gt_tokens, padding_tokens], dim=1)

        # Calculate the cross entropy loss - transpose the 1 and 2 dimensions because we want a loss value per token not per vocab
        loss = loss_fn(gllm_probs.transpose(1, 2), gt_probs.transpose(1, 2))

        return loss


def train_loop(
    model: GLLM,
    dataloader: DataLoader,
    network_desc: str,
    n_epochs: int = 10,
    loss_fn: Literal["cosine", "crossentropy"] = "crossentropy",
    checkpoint_every: int = 10,
):

    model.train(mode=True)
    timestamp = datetime.now().strftime("%H-%M-%d-%m")
    train_losses = []

    for epoch in range(n_epochs):
        gllm_responses = []

        for batch in dataloader:
            model.optimizer.zero_grad()
            questions = batch["questions"]
            gllm_prompts = model.build_prompts(questions=questions, network_desc=network_desc, model="gllm")

            gllm_tokens, gllm_logprobs, gllm_hidden_states = model(batch["graphs"], gllm_prompts)
            txt_response = model.llm.tokenizer.batch_decode(gllm_tokens, skip_special_tokens=False)[-1]
            gllm_responses.append(txt_response)
            if loss_fn == "cosine":
                loss = model.get_cosine_embeddings_loss(batch["gt_hidden_states"], gllm_hidden_states)
            elif loss_fn == "crossentropy":

                loss = model.get_crossentropy_loss(
                    gt_probs=batch["gt_logprobs"],
                    gt_tokens=batch["gt_tokens"],
                    gllm_probs=gllm_logprobs,
                    gllm_tokens=gllm_tokens,
                )
            # Do the backwards pass
            loss.backward()

            # Clear up any gpu memory that may be holding onto tensors unnecessarily
            torch.cuda.empty_cache()

            model.optimizer.step()

        if epoch % checkpoint_every == 0:
            train_losses.append(loss.item())
            plt.plot(train_losses, label="Train Loss")
            plt.title("Cross Entropy Loss")
            plt.ylabel("Loss")
            plt.xlabel("Epoch")
            print(f"{batch['gt_answers'][-1]}\n{txt_response}")

            if not os.path.exists(f"./Trainings/{timestamp}"):
                os.makedirs(f"./Trainings/{timestamp}")
            plt.savefig(f"./Trainings/{timestamp}/Training_Losses.png")
            torch.save(model.state_dict(), f"./Trainings/{timestamp}/{epoch+1}.pth")
