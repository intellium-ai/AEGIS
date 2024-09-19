import json
import os
from typing import List, Tuple, Dict, Generator, Optional
from functools import cached_property
import warnings

import torch
from torch.distributions import Categorical

from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
from peft.tuners.lora import LoraConfig
from peft import PeftModelForCausalLM, PeftModel, PeftConfig


DEFAULT_DEVICE_MAP: dict = {
    'model.embed_tokens': 0,
    'lm_head': 0,
    'model.layers.0': 0,
    'model.layers.1': 1,
    'model.layers.2': 1,
    'model.layers.3': 1,
    'model.layers.4': 1,
    'model.layers.5': 1,
    'model.layers.6': 1,
    'model.layers.7': 1,
    'model.layers.8': 1,
    'model.layers.9': 2,
    'model.layers.10': 2,
    'model.layers.11': 2,
    'model.layers.12': 2,
    'model.layers.13': 2,
    'model.layers.14': 2,
    'model.layers.15': 2,
    'model.layers.16': 2,
    'model.layers.17': 3,
    'model.layers.18': 3,
    'model.layers.19': 3,
    'model.layers.20': 3,
    'model.layers.21': 3,
    'model.layers.22': 3,
    'model.layers.23': 3,
    'model.norm': 3,
}


class LLM(torch.nn.Module):
    def __init__(
        self,
        model_name_or_path = "HuggingFaceTB/SmolLM-1.7B-Instruct",
        peft_config: LoraConfig = None,
        device_map: dict = DEFAULT_DEVICE_MAP
    ):
        super().__init__()

        # Initialise quantisation config
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16,
            device_map=device_map,
            quantization_config=self.bnb_config,
            attn_implementation="sdpa"
        )

        self.model: PeftModelForCausalLM = PeftModelForCausalLM(
            model=model,
            peft_config=peft_config
        )

        self.device_map = device_map

        self.model.gradient_checkpointing_enable()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, torch_dtype=torch.float16, padding=True, device_map="auto", padding_side="right"
        )

        print(self.get_n_trainable_llm_parameters())

        # Figure out what tokens to allow in action generate calls
        self.numeric_token_ids, self.non_numeric_token_ids = self._get_filtered_tokenizer_vocab()

        # Get start and end graph tag embeddings
        self.graph_start_inputs = self.get_input_embeddings(prompts=["<graph>"])
        self.graph_end_inputs = self.get_input_embeddings(prompts=["</graph>"])
        self.close_msg_inputs = self.get_input_embeddings(
            prompts=[self.tokenizer.eos_token + "\n" + self.tokenizer.bos_token + "assistant\n"]
        )
        self.start_msg_inputs = self.get_input_embeddings(prompts=[self.tokenizer.bos_token + "user\n"])
        self.pad_token_inputs = self.get_input_embeddings(prompts=[self.tokenizer.pad_token])
        # Hacky way to get the size of each tokens embedding - this helps us to align the GNN and LLM output shapes later.
        self.llm_embedding_size = self.get_input_embeddings(prompts=["hack"])["inputs_embeds"].shape[2]

        self.padding_logit = torch.zeros([len(self.tokenizer)], device=self.device, dtype=torch.float32)
        self.padding_logit[self.tokenizer.pad_token_id] = (
            100.0
        )
        self.pad_token = torch.tensor([self.tokenizer.pad_token_id], device=self.device, dtype=torch.int32)

    def get_n_trainable_llm_parameters(self) -> str:
        trainable_model_params = 0
        all_model_params = 0
        for _, param in self.named_parameters():
            all_model_params += param.numel()
            if param.requires_grad:
                trainable_model_params += param.numel()
        return f"Trainable LLM parameters: {trainable_model_params:,}\nAll LLM parameters: {all_model_params:,}\nPercentage of trainable LLM parameters: {trainable_model_params / all_model_params:.2%}%"

    def _get_filtered_tokenizer_vocab(self) -> Tuple[Dict[str, int], Dict[str, int]]:

        # Allowed tokens:
        allowed_tokens = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "."]

        tokenizer_vocab = self.tokenizer.get_vocab()
        numeric_token_ids = {}
        non_numeric_token_ids = {}
        for k, v in tokenizer_vocab.items():
            if k in allowed_tokens:
                # Allowed token
                if k in allowed_tokens[:-1]:
                    # numeric
                    numeric_token_ids[k] = v
                elif k in allowed_tokens[-1]:
                    # non-numeric (can only be a '.')
                    non_numeric_token_ids[k] = v

        return numeric_token_ids, non_numeric_token_ids

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
        batch_size = graph_embs.shape[0]
        return torch.cat(
            [
                self.graph_start_inputs["inputs_embeds"].expand(batch_size, -1, -1),
                graph_embs,
                self.graph_end_inputs["inputs_embeds"].expand(batch_size, -1, -1),
            ],
            dim=1,
        )

    def format_inputs(self, inputs: torch.Tensor, graph_embeddings: torch.Tensor) -> torch.Tensor:
        assert inputs["inputs_embeds"].shape[0] == graph_embeddings.shape[0]
        batch_size = inputs["inputs_embeds"].shape[0]
        original_len = inputs["inputs_embeds"].shape[1]

        embeds = inputs["inputs_embeds"]
        attention_mask = inputs["attention_mask"]
        # Concatenate all sequences up to the text prompt embeddings
        embeds = torch.cat(
            [
                self.start_msg_inputs["inputs_embeds"].expand(batch_size, -1, -1),
                graph_embeddings,
                embeds,
            ],
            dim=1,
        )

        # How many new tokens did we add to the beginning of the sequence?
        n_tokens = embeds.shape[1] - original_len
        attention_mask = torch.cat([torch.ones((batch_size, n_tokens), device=self.device), attention_mask], dim=1)

        # Expand all sequences to have minimum of close_msg_inputs tokens as slack
        n_tokens = self.close_msg_inputs["inputs_embeds"].shape[1]
        new_pad_embeds = self.pad_token_inputs["inputs_embeds"].repeat(batch_size, n_tokens, 1)
        new_pad_attns = torch.zeros((batch_size, n_tokens), device=self.device)
        embeds = torch.cat([embeds, new_pad_embeds], dim=1)
        attention_mask = torch.cat([attention_mask, new_pad_attns], dim=1)

        # Replace N most left pad tokens with close_msg_inputs embeds and swap corresponding attentions to 1s
        close_msg_embeds = self.close_msg_inputs["inputs_embeds"].repeat(batch_size, 1, 1)

        for i in range(batch_size):
            pad_positions = (attention_mask[i] == 0).nonzero(as_tuple=True)[0][:n_tokens]

            # Replace padding token embeddings with close_msg_embeds embeddings
            embeds[i, pad_positions] = close_msg_embeds[i]

            # Update attention masks to 1 at these positions
            attention_mask[i, pad_positions] = 1

        inputs["inputs_embeds"] = embeds
        inputs["attention_mask"] = attention_mask
        return inputs
    
    @cached_property
    def _embeddings_layer(self):
        return self.model.get_input_embeddings()

    def get_input_embeddings(
        self,
        prompts: List[str] = None,
        token_ids: List[int] = None,
        grad: bool = False,
    ) -> Dict[str, Tuple[torch.Tensor, List[int]]]:
        """Your non-standard .generate"""

        assert (
            prompts is not None or token_ids is not None
        ), "A text prompt or list of token ids must be passed to get_embeddings"

        if prompts:
            inputs = self.tokenizer.batch_encode_plus(prompts, return_tensors="pt", padding=True)
        else:
            inputs = {"input_ids": token_ids, "attention_mask": torch.ones(len(token_ids), dtype=torch.bool, device=self.device)}

        inputs['input_ids'] = inputs['input_ids'].to(self.device)
        inputs['attention_mask'] = inputs['attention_mask'].to(self.device)


        if not grad:
            with torch.no_grad():
                inputs["inputs_embeds"] = self._embeddings_layer(inputs["input_ids"])
        else:
            inputs["inputs_embeds"] = self._embeddings_layer(
                inputs["input_ids"]
            ).requires_grad_(True)

        # Remove input ids as as key since no longer needed and a liability to keep updated
        del inputs["input_ids"]
        return inputs

    def get_output_embeddings(self, inputs_embeds: torch.Tensor, grad=True):
        if grad:
            last_hidden_states = self.model.forward(inputs_embeds=inputs_embeds, output_hidden_states=True)[1][
                "hidden_states"
            ][-1][:, -1:, :]
        else:
            with torch.no_grad():
                last_hidden_states = self.model.forward(inputs_embeds=inputs_embeds, output_hidden_states=True)[1][
                    "hidden_states"
                ][-1][:, -1:, :]

        return last_hidden_states

    def generate_from_embeddings(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        grad: bool = True,
        max_new_tokens: int = 400,
        restrict_output: bool = True,
        last_hidden_state: bool = False,
        do_sample: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = inputs_embeds.device

        next_token_ids = torch.empty(0, dtype=torch.int32, device=device)
        next_token_logits = torch.empty(0, dtype=torch.float32, device=device)

        n_tokens = 0
        stop_generating = False
        per_token_attention = torch.ones((inputs_embeds.shape[0], 1), device=device, dtype=torch.bool)
        attention_mask = attention_mask.to(torch.bool)

        # Generate until max_new_tokens reached
        sequences_terminated = torch.zeros((inputs_embeds.size(0)), dtype=torch.bool, device=device)

        while not stop_generating:
            if not grad:
                with torch.no_grad():
                    output = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
            else:
                output = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)

            # Figure out which logits in the sequence to sample from (the right-most significant token determined with the attention mask)
            last_one_positions = torch.flip(attention_mask, dims=[1]).cumsum(dim=1).eq(1).max(dim=1)[1]
            last_non_pad_indices = attention_mask.size(1) - 1 - last_one_positions

            # Get next logits for each input sequence
            logits = output.logits[torch.arange(inputs_embeds.size(0), device=device), last_non_pad_indices].unsqueeze(1)

            # Apply restriction on the output logits (or don't)
            if restrict_output:
                if next_token_ids.numel() != 0:
                    last_token_ids = next_token_ids[:, -1].tolist()
                else:
                    last_token_ids = None
                token_ids, logits = self._filter_logits(
                    logits=logits, prev_token_ids=last_token_ids, do_sample=do_sample
                )  # WARNING: probs here is ONLY the top logprob
            else:
                if do_sample:
                    dist = Categorical(logits=logits)
                    token_ids = dist.sample()
                else:
                    token_ids = torch.argmax(logits, dim=-1).to(torch.int32)

            # For any sequence that is done already, just set their next token and prob to the last token and prob (the EOS token and prob)
            if torch.any(sequences_terminated):
                # For sequences that are done, set their next token and prob to pad token and last token prob
                token_ids[sequences_terminated] = self.pad_token
                logits[sequences_terminated] = self.padding_logit #next_token_logits[eos_mask, -1, :].unsqueeze(1)

            # Add the new tokens to the sequences
            next_token_ids = torch.cat([next_token_ids, token_ids], dim=1)
            next_token_logits = torch.cat([next_token_logits, logits], dim=1)

            # Check for EOS token IDs so we can exclude that sequence from the next token generation.
            sequences_terminated = torch.logical_or(sequences_terminated, token_ids.squeeze() == self.tokenizer.eos_token_id)

            # Get embeddings of the new tokens
            new_embeddings = self.get_input_embeddings(token_ids=token_ids, grad=True)["inputs_embeds"]

            # Add the new token to the inputs_embeds and expand the attention mask accordingly
            # TODO: For sequences that are complete, don't generate more tokens for them to save on compute (they get binned anyway when replaced with EOS)
            inputs_embeds = torch.cat([inputs_embeds, new_embeddings], dim=1)
            attention_mask = torch.cat([attention_mask, per_token_attention], dim=1)

            n_tokens += 1

            # Check if all sequences have reached their eos token or max_new_tokens limit reached
            if n_tokens == max_new_tokens or torch.sum(sequences_terminated) == inputs_embeds.shape[0]:
                stop_generating = True

        # Collect the last hidden states
        if last_hidden_state:
            # Get the output hidden states of only the response
            last_hidden_state = self.model.forward(
                inputs_embeds=inputs_embeds, attention_mask=attention_mask, output_hidden_states=True
            )[1]["hidden_states"][-1][:, -1:, :]
            return next_token_ids, next_token_logits, last_hidden_state

        return next_token_ids, next_token_logits
    
    def yield_probs_given_tokens(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        token_ids: torch.Tensor
    ) -> Generator[torch.Tensor, None, None]:
        
        warnings.warn('yield_probs_given_tokens has not been tested since a substantial update to the LLM class!\nIt may not work as intended!')

        for i, token_idx in enumerate(token_ids):
            output = self.model.forward(inputs_embeds=inputs_embeds, attention_mask=attention_mask)

            # Figure out which logits in the sequence to sample from (the right-most significant token determined with the attention mask)
            last_one_positions = torch.flip(attention_mask, dims=[1]).cumsum(dim=1).eq(1).max(dim=1)[1]
            last_non_pad_indices = attention_mask.size(1) - 1 - last_one_positions

            # Get next logits for each input sequence
            logits = output.logits[torch.arange(inputs_embeds.size(0)), last_non_pad_indices]

            prev_token_ids = None if i == 0 else token_ids[:i]
            allowed_indices = self._generate_allowed_indicies(logits, prev_token_ids)

            filtered_logits = logits[:, allowed_indices[0]]
            filtered_probs = torch.nn.functional.softmax(filtered_logits, dim=1)

            local_idx = torch.where(allowed_indices[0] == token_idx.to(allowed_indices[0].device))[0]
            probs = filtered_probs.squeeze()[local_idx]

            # Get embeddings of the new tokens
            new_embeddings = self.get_input_embeddings(token_ids=token_idx.reshape((1,1)), grad=True)["inputs_embeds"]

            # Add the new token to the inputs_embeds and expand the attention mask accordingly
            inputs_embeds = torch.cat([inputs_embeds, new_embeddings], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones((1,1), device=attention_mask.device)], dim=1)

            yield probs

    def _filter_logits(self, logits: torch.Tensor, prev_token_ids: List[int], do_sample: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Filter the logits to only allow the allowed tokens and to ensure it follows the required format for primaite action taking."""
        allowed_indices = self._generate_allowed_indicies(logits, prev_token_ids)

        warnings.warn('_filter_logits has not been tested since a substantial update to the LLM class!\nIt may not work as intended!')


        token_ids = torch.empty(logits.shape[0], dtype=torch.int32, device=logits.device)
        logits = torch.empty(logits.shape[0], dtype=torch.float32, device=logits.device)

        # For each sequence, apply the filtering to the output logits and select the highest logit
        for i in range(logits.shape[0]):
            filtered_logits = logits[i, allowed_indices[i]]

            if do_sample:
                dist = Categorical(logits=logits)
                sampled_indicies = dist.sample()

                token_ids[i] = allowed_indices[i][sampled_indicies]
                logits[i] = filtered_logits[sampled_indicies]
            else:
                max_logits, max_indices = torch.max(filtered_logits, dim=-1)
                token_ids[i] = allowed_indices[i][max_indices]
                logits[i] = max_logits

        return token_ids.unsqueeze(1), logits.unsqueeze(1)

    def _generate_allowed_indicies(self, logits: torch.Tensor, prev_token_ids: Optional[List[int]]) -> List[torch.Tensor]:
        allowed_indices = []
        if prev_token_ids is not None:
            for prev_token_id in prev_token_ids:
                # If the previous was not a number - force one next
                if not prev_token_id in self.numeric_token_ids.values():
                    filtered_vocab_set = set(self.numeric_token_ids.values())

                # If the previous was a number, then it can be either a '.' or any number (0-9)
                else:
                    filtered_vocab_set = set(self.numeric_token_ids.values()).union(self.non_numeric_token_ids.values())

                # Use torch.long cos we use this as an index later
                allowed_indices.append(
                    torch.tensor(
                        [idx for idx in range(logits.shape[-1]) if idx in filtered_vocab_set], dtype=torch.int32, device=logits.device
                    )
                )

        # If it's the first token, force a numeric first token
        else:
            # All sequences can start with any number (0-9) but NOT a '.'
            filtered_vocab_set = set(self.numeric_token_ids.values())

            for _ in range(logits.shape[0]):
                allowed_indices.append(
                    torch.tensor(
                        [idx for idx in range(logits.shape[-1]) if idx in filtered_vocab_set], dtype=torch.int32, device=logits.device
                    )
                )
                
        return allowed_indices
    
    def _generate_last_state(
        self, 
        texts: List[str] = None, 
        input_ids: torch.Tensor = None, 
        input_embeddings: torch.Tensor = None
    ) -> torch.Tensor:
        
        warnings.warn('_generate_last_state has not been tested since a substantial update to the LLM class!\nIt may not work as intended!')


        if texts is not None:
            texts = [self.tokenizer.bos_token + text + self.tokenizer.eos_token for text in texts]
            tokenizer_output = self.tokenizer(texts, return_tensors="pt", padding=True)

            input_ids = tokenizer_output["input_ids"]
            eos_idx = torch.sum(tokenizer_output["attention_mask"], dim=1) - 1

        if input_ids is not None:
            input_embeddings = self.model.pretrained_model.get_input_embeddings()(input_ids)

        hidden_layer_output = self.model(inputs_embeds=input_embeddings)[1]["hidden_states"][-1]

        if texts is None:
            eos_idx = torch.full((hidden_layer_output.shape[0],), fill_value=-1)
        final_embeddings = hidden_layer_output[
            torch.arange(
                hidden_layer_output.shape[0],
            ),
            eos_idx,
        ]

        return final_embeddings

    def generate_last_state(self, grad: bool = True, **kwargs) -> torch.Tensor:
        if grad:
            return self._generate_last_state(**kwargs)
        else:
            with torch.no_grad():
                return self._generate_last_state(**kwargs)

    def save(self, path: str) -> None:
        # Save Tokenizer
        self.tokenizer.init_kwargs.pop('torch_dtype', None)
        self.tokenizer.save_pretrained(path)
        # Save LLM
        self.model.save_pretrained(path)
        init_kwargs = {
            "device_map": self.device_map,
        }

        json.dump(init_kwargs, open(os.path.join(path, 'llm_init_kwargs.json'), 'w'))

    @classmethod
    def load(cls, path: str):
        # TODO: Is there a nicer way to do this? :)
        adapter_config = json.load(open(os.path.join(path, 'adapter_config.json')))
        peft_config = PeftConfig.from_peft_type(**adapter_config)
        try:
            init_kwargs = json.load(open(os.path.join(path, 'llm_init_kwargs.json')))
            llm = cls(device_map=init_kwargs['device_map'], peft_config=peft_config)
            device_map = init_kwargs['device_map']
        except:
            init_kwargs = {}
            device_map = DEFAULT_DEVICE_MAP
            llm = cls(peft_config=peft_config, device_map=device_map)
        
        base_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=adapter_config['base_model_name_or_path'], quantization_config=llm.bnb_config, torch_dtype=torch.float16, attn_implementation="sdpa", device_map=device_map)
        print(llm.bnb_config)
        peft_model = PeftModelForCausalLM.from_pretrained(base_model, path, is_trainable=True)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=path, torch_dtype=torch.float16, padding=True, device_map='auto', padding_side='right')
        
        llm.model = peft_model
        llm.tokenizer = tokenizer
        llm.model.gradient_checkpointing_enable()
        return llm
    
    @cached_property
    def device(self):
        return f"cuda:{self.device_map['model.embed_tokens']}"