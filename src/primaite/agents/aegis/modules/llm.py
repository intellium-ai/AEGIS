import torch
from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from peft.tuners.lora import LoraConfig
from typing import List, Tuple, Dict


class LLM(torch.nn.Module):
    def __init__(
        self,
        model_name="HuggingFaceTB/SmolLM-1.7B-Instruct",
        peft_config: LoraConfig = None,
    ):
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
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
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

        # Hacky way to get the size of each tokens embedding - this helps us to align the GNN and LLM output shapes later.
        self.llm_embedding_size = self.get_embeddings(prompt="hack", apply_chat_tokens=False).shape[2]

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
        self,
        prompt: str = None,
        token_ids: List[int] = None,
        system: str = None,
        apply_chat_tokens: bool = True,
        grad: bool = False,
    ) -> torch.Tensor:
        """Your non-standard .generate"""
        assert (
            prompt is not None or token_ids is not None
        ), "A text prompt or list of token ids must be passed to get_embeddings"
        if prompt:
            if apply_chat_tokens:
                inputs = self.apply_chat_template(system=system, prompt=prompt, close_usr_msg=False)
            elif not apply_chat_tokens and not system:
                inputs = prompt
            inputs = self.tokenizer.encode(inputs, return_tensors="pt")
        else:
            inputs = token_ids
        if not grad:
            with torch.no_grad():
                embs = self.model.pretrained_model.get_input_embeddings()(inputs)
        else:
            embs = self.model.pretrained_model.get_input_embeddings()(inputs).requires_grad_(True)
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
            logits = next_token_embs[0][:, -1, :]

            # Apply restriction on the output logits (or don't)
            if restrict_output:
                token_id, prob = self._filter_logits(
                    logits=logits, prev_token_id=next_token_ids[-1] if next_token_ids else None
                )
            else:
                logit = torch.max(logits, dim=-1)
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
    
    def _generate_last_state(
        self, 
        texts: List[str] = None, 
        input_ids: torch.Tensor = None, 
        input_embeddings: torch.Tensor = None
    ) -> torch.Tensor:

        if texts is not None:
            texts = [self.tokenizer.bos_token + text + self.tokenizer.eos_token for text in texts]
            tokenizer_output = self.tokenizer(texts, return_tensors='pt', padding=True)

            input_ids = tokenizer_output['input_ids']
            eos_idx = torch.sum(tokenizer_output['attention_mask'], dim=1) - 1

        if input_ids is not None:
            input_embeddings = self.model.pretrained_model.get_input_embeddings()(input_ids)

        hidden_layer_output = self.model(inputs_embeds=input_embeddings)[1]['hidden_states'][-1]

        if texts is None:
            eos_idx = torch.full((hidden_layer_output.shape[0], ), fill_value=-1)
        final_embeddings = hidden_layer_output[torch.arange(hidden_layer_output.shape[0], ), eos_idx]

        return final_embeddings

    def generate_last_state(
        self, 
        grad: bool = True,
        **kwargs
    ) -> torch.Tensor:
        
        if grad:
            return self._generate_last_state(**kwargs)
        else:
            with torch.no_grad():
                return self._generate_last_state(**kwargs)