import logging
from typing import List, Literal, Tuple
from datetime import datetime
import os
from itertools import zip_longest
from cProfile import Profile
import warnings

from tqdm.auto import trange, tqdm
import torch
from transformers import Adafactor
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from torch.utils.tensorboard import SummaryWriter
from peft.tuners.lora import LoraConfig
import numpy as np

from primaite.agents.aegis.modules.llm import LLM
from primaite.agents.aegis.modules.ge import GraphEmbedding
from primaite.agents.aegis.prompts import LLM_REASONING_PROMPT

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
        loss_fn: Literal["cosine", "crossentropy"] = "crossentropy",
        do_tensorboard_logging: bool = True
    ):
        super(GLLM, self).__init__()

        self.ge_device = ge_device
        self.model_name = model_name
        self.learning_rate = learning_rate

        self.llm = LLM(
            model_name_or_path = self.model_name, 
            peft_config = peft_config
        )

        self.ge = GraphEmbedding(
            device = self.ge_device,
            in_channels = 6,
            hidden_dim = 256,
            n_tokens = 100,
            n_gat_layers = 1,
            output_dim = self.llm.llm_embedding_size,
        ).to(self.ge_device)

        self.optimizer = Adafactor(self.parameters(), lr=self.learning_rate, relative_step=False)
        self.default_lr = learning_rate
        
        self.do_tensorboard_logging = do_tensorboard_logging
        if do_tensorboard_logging:
            self.writer = SummaryWriter(flush_secs=15)
            self.global_step = 0
            self.global_epoch = 0


        if loss_fn ==  "crossentropy":
            self.loss_fn = self._crossentropy_loss
        elif loss_fn == "cosine":
            warnings.warn('_cosine_embeddings_loss is deprecated! This functionality might not work correctly!', DeprecationWarning)
            self.loss_fn = self._cosine_embeddings_loss
        else:
            raise NotImplementedError

    def forward(self, graph_batch, gllm_prompts) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """This should return the action integer"""
        # 1.0 - Get the graph token(s)
        graph_embs = self.ge(graph_batch).to(torch.float16)
        graph_embs = graph_embs.to(self.llm.embedding_device)
        graph_embs = self.llm._concat_graph_tags(graph_embs)  # Add <graph>...</graph>

        # 2.0 - Get the combined embeddings of graph and text tokens.
        inputs = self.llm.get_input_embeddings(prompts=gllm_prompts, grad=False)

        # 3.0 - Add BOS, graph token embs, text token embs and EOS token - shifting padding along to the right as required to maintain standardization of sequence length.
        inputs = self.llm.format_inputs(inputs=inputs, graph_embeddings=graph_embs)

        # 3.0 - Generate the response
        token_ids, gllm_logits = self.llm.generate_from_embeddings(
            inputs_embeds=inputs["inputs_embeds"].to(self.llm.embedding_device),
            attention_mask=inputs["attention_mask"].to(self.llm.embedding_device),
            max_new_tokens=150,
            restrict_output=False,
            grad=True,
            last_hidden_state=False,
            do_sample=False
        )

        return token_ids, gllm_logits

    def build_prompts(self, data: Batch) -> List[str]:
        warnings.warn('build_prompts has not been changed to include the target action text.')

        return [LLM_REASONING_PROMPT.format(
            node_ids = item.node_ids,
            services = item.services,
            node_services = item.node_services,
            obs_act_history = item.obs_act_history if hasattr(item, 'obs_act_history') else None,
            current_obs_diff = item .current_obs_diff if hasattr(item, 'current_obs_diff') else None,
        ) for item in data.to_data_list()]
    
    def _get_target_logits_and_tokens(self, texts: List[str], device: torch.device = 'cuda:0'):
        # Construct mock logprobs for gt repsonse
        gt_tokens = self.llm.tokenizer.batch_encode_plus(texts, return_tensors="pt", padding=True)["input_ids"].to(device)

        # [batch_size, seq_length, vocab_size]
        gt_logits = torch.zeros((gt_tokens.shape[0], gt_tokens.shape[1], len(self.llm.tokenizer)), dtype=torch.float16, device=device)
        gt_logits.scatter_(2, gt_tokens.unsqueeze(-1), 100.0)

        return gt_tokens, gt_logits

    def _cosine_embeddings_loss(self, batch: Batch, gllm_hidden_states: torch.Tensor, *args, **kwargs):
        openai_hidden_states = batch.openai_hidden_states

        loss_fn = torch.nn.CosineEmbeddingLoss()
        loss = loss_fn(
            openai_hidden_states.view(-1, self.llm.llm_embedding_size).to(gllm_hidden_states.device),
            gllm_hidden_states.view(-1, self.llm.llm_embedding_size),
            torch.ones(gllm_hidden_states.shape[0]).to(gllm_hidden_states.device),
        )
        return loss

    def _crossentropy_loss(
            self, 
            gt_logits: torch.Tensor, 
            gllm_logits: torch.Tensor, 
            *args, **kwargs
    ) -> torch.Tensor:
        padding_logit = torch.zeros([len(self.llm.tokenizer)], device=gllm_logits.device, dtype=torch.float32)
        padding_logit[self.llm.tokenizer.pad_token_id] = (
            100.0 
        )
        # Pad the shortest set of tensors if required
        batch_size = gllm_logits.shape[0]
        if gt_logits.shape[1] > gllm_logits.shape[1]:
            # gt sequences are longer - truncate the gt responses
            gt_logits = gt_logits[:, : gllm_logits.shape[1], :]

        elif gllm_logits.shape[1] > gt_logits.shape[1]:
            # gllm sequences are longer - pad openai responses
            n_pads = gllm_logits.shape[1] - gt_logits.shape[1]
            padding_logits = padding_logit.repeat(batch_size, n_pads, 1)

            gt_logits = torch.cat([gt_logits, padding_logits], dim=1)

        # Calculate the cross entropy loss - transpose the 1 and 2 dimensions because we want a loss value per token not per vocab
        return torch.nn.functional.cross_entropy(gllm_logits.transpose(1, 2), gt_logits.transpose(1, 2))

    def train_model(
        self,
        train_dataloader: DataLoader,
        n_epochs: int = 10,
        lr: float = None,
        gradient_accumulation_steps: int = 1,
        save_every_n_epochs: int = None,
        progress_bar: bool = True,
        do_profile: bool = False,
    ):
        self.train()
        run_timestamp = datetime.now().strftime("%H-%M-%d-%m")

        # Set/Reset Run LR
        if lr:
            self.set_optimizer_lr(lr)
        else:
            self.reset_optimizer_lr()
        self.optimizer.zero_grad()

        # Main training loop
        for epoch in (pbar := trange(n_epochs, desc='Epoch', unit='epoch', disable=not progress_bar)):
            epoch_train_loss = 0
            epoch_text_lengths = []

            tokens_correct, tokens_total = 0, 0

            for batch in tqdm(train_dataloader, desc='Batch', leave=False, unit='batch', disable=not progress_bar):
                if do_profile:
                    profiler = Profile()
                    profiler.enable()

                # Build LLM text inputs
                prompts = self.build_prompts(data=batch)

                # Put through model and get ground truths from text
                gllm_tokens, gllm_logits = self(batch, prompts)
                
                gt_tokens, gt_logits = self._get_target_logits_and_tokens(batch.reasoning)
                gt_logits, gt_tokens = gt_logits.to(gllm_logits.device), gt_tokens.to(gllm_logits.device)

                # Caculate number of tokes correctly predicted / total
                step_tokens_correct, step_tokens_total = num_tokens_correct_total(gllm_tokens.detach(), gt_tokens)
                tokens_correct += step_tokens_correct
                tokens_total += step_tokens_total

                # Decode text reponses for inspection
                txt_responses = self.llm.tokenizer.batch_decode(gllm_tokens.detach(), skip_special_tokens=False)
                torch.cuda.empty_cache()

                # Calc loss
                loss = self.loss_fn(
                    gt_logits=gt_logits,
                    gllm_logits=gllm_logits,
                )

                loss.backward()
                torch.cuda.empty_cache()

                if self.global_step % gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # Log Step Metrics
                if self.do_tensorboard_logging:
                    self.writer.add_scalar('step/train_loss', loss.detach().cpu().item(), global_step=self.global_step)
                    self.writer.add_text('step/text_response', txt_responses[-1], global_step=self.global_step)
                epoch_text_lengths += [len(response) for response in txt_responses]
                epoch_train_loss += loss.detach().cpu().item() * batch.batch_size

                self.global_step += 1

                if do_profile:
                    profiler.disable()
                    profiler.dump_stats('profiler_stats')
                    profiler.print_stats()
                    exit()

            # Log Epoch Metrics
            epoch_train_loss /= len(train_dataloader.dataset)

            if self.do_tensorboard_logging:
                self.writer.add_scalar('epoch/train_loss', epoch_train_loss, global_step=self.global_epoch)
                self.writer.add_scalar('epoch/train_tok_acc', tokens_correct / tokens_total, global_step=self.global_epoch)
                self.writer.add_histogram('epoch/reasoning_lengths', np.array(epoch_text_lengths), global_step=self.global_epoch)

            pbar.set_description(f'Train Loss: {epoch_train_loss:.4f}')

            # Save model if required
            if save_every_n_epochs:
                if self.global_epoch % save_every_n_epochs == 0:
                    run_name = self.writer.get_logdir() if self.do_tensorboard_logging else run_timestamp
                    self.save(os.path.join(os.curdir, '/runs/', run_name, f'epoch_{self.global_epoch}'))

            self.global_epoch += 1

    @classmethod
    def load(cls, path):
        raise NotImplementedError

    def save(self, path: str) -> None:
        self.llm.save(path)
        self.ge.save(path)
        torch.save(self.optimizer.state_dict(), os.path.join(path, 'git_agent_optim_state.pt'))


    def reset_optimizer_lr(self) -> None:
        for group in self.optimizer.param_groups:
            group['lr'] = self.default_lr

    def set_optimizer_lr(self, new_lr: float) -> None:
        for group in self.optimizer.param_groups:
            group['lr'] = new_lr
        
    def get_optimizer_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']


def num_tokens_correct_total(tokens1: torch.Tensor, tokens2: torch.Tensor, pad_token_id: int = 2):
    total_valid_tokens = 0
    total_correct = 0

    for seq1, seq2 in zip(tokens1, tokens2):
        for tok1, tok2 in zip_longest(seq1, seq2):
            if tok1 is not None and tok2 is not None:
                if tok1 == tok2:
                    if tok1 != pad_token_id:
                        total_valid_tokens += 1
                        total_correct += 1
                else:
                    total_valid_tokens += 1
            elif (tok1 is None) or (tok2 is None):
                total_valid_tokens += 1

    return total_correct, total_valid_tokens