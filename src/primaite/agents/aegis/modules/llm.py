import torch
from transformers import BitsAndBytesConfig, AutoTokenizer
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
        self.numeric_token_ids, self.non_numeric_token_ids = self._get_filtered_tokenizer_vocab()

        # Get start and end graph tag embeddings
        self.graph_start_emb = self.get_input_embeddings(prompts=["<graph>"])
        self.graph_end_emb = self.get_input_embeddings(prompts=["</graph>"])
        self.close_msg_emb = self.get_input_embeddings(
            prompts=[self.tokenizer.eos_token + "\n" + self.tokenizer.bos_token + "assistant\n"]
        )
        self.start_msg_emb = self.get_input_embeddings(prompts=[self.tokenizer.bos_token + "user\n"])

        # Hacky way to get the size of each tokens embedding - this helps us to align the GNN and LLM output shapes later.
        self.llm_embedding_size = self.get_input_embeddings(prompts=["hack"]).shape[2]

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
                self.graph_start_emb.expand(batch_size, -1, -1),
                graph_embs,
                self.graph_end_emb.expand(batch_size, -1, -1),
            ],
            dim=1,
        )

    def format_embeddings(self, text_embeddings: torch.Tensor, graph_embeddings: torch.Tensor) -> torch.Tensor:
        assert text_embeddings.shape[0] == graph_embeddings.shape[0]
        batch_size = text_embeddings.shape[0]

        # Concatenate together everything
        embeddings = torch.cat(
            [
                self.start_msg_emb.expand(batch_size, -1, -1),
                graph_embeddings,
                text_embeddings,
                self.close_msg_emb.expand(batch_size, -1, -1),
            ],
            dim=1,
        )

        return embeddings

    def get_input_embeddings(
        self,
        prompts: List[str] = None,
        token_ids: List[int] = None,
        grad: bool = False,
    ) -> torch.Tensor:
        """Your non-standard .generate"""
        assert (
            prompts is not None or token_ids is not None
        ), "A text prompt or list of token ids must be passed to get_embeddings"
        if prompts:
            inputs = self.tokenizer.batch_encode_plus(prompts, return_tensors="pt", padding=True).input_ids
        else:
            inputs = token_ids

        if not grad:
            with torch.no_grad():
                embs = self.model.pretrained_model.get_input_embeddings()(inputs)
        else:
            embs = self.model.pretrained_model.get_input_embeddings()(inputs).requires_grad_(True)
        return embs

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
        text_embeddings: torch.Tensor,
        grad: bool = True,
        max_new_tokens: int = 10,
        restrict_output: bool = True,
        last_hidden_state=False,
    ) -> Tuple[List[int], torch.Tensor]:

        next_token_ids = torch.empty(0, dtype=torch.int8)
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

            # Get next logits for each input sequence
            logits = next_token_embs[1]["logits"][:, -1]

            # Apply restriction on the output logits (or don't)
            if restrict_output:
                if next_token_ids.numel() != 0:
                    last_token_ids = next_token_ids[:, -1].tolist()
                    print("last tokens:", last_token_ids)
                else:
                    last_token_ids = None
                token_ids, probs = self._filter_logits(logits=logits, prev_token_ids=last_token_ids)
            else:
                logits = torch.max(logits, dim=-1)
                token_ids = logits.indices.unsqueeze(1)
                probs = logits.values.unsqueeze(1)

            # Check what device the output is on
            device = probs.device

            # Update probs and token ids
            next_token_ids = torch.cat([next_token_ids.to(device), token_ids], dim=1)
            next_token_probs = torch.cat([next_token_probs.to(device), probs], dim=1)

            # Get embeddings of the new token and add it to the previous embeddings
            new_embeddings = self.get_input_embeddings(token_ids=token_ids, grad=True)

            text_embeddings = torch.cat([text_embeddings, new_embeddings], dim=1)
            n_tokens += 1

            # Check for eos token or max_new_tokens limit reached
            # TODO: Add back EOS checker and pad all finished sequences until all are finished generating.
            if n_tokens == max_new_tokens:
                stop_generating = True

        # Collect the last hidden states
        if last_hidden_state:
            # Get the output hidden states of only the response
            last_hidden_state = self.model.forward(inputs_embeds=text_embeddings, output_hidden_states=True)[1][
                "hidden_states"
            ][-1][:, -1:, :]
            return next_token_ids, next_token_probs, last_hidden_state

        return next_token_ids, next_token_probs

    def _filter_logits(self, logits: torch.Tensor, prev_token_ids: List[int]) -> Tuple[int, torch.Tensor]:
        allowed_indices = []

        if prev_token_ids:
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
                        [idx for idx in range(logits.shape[-1]) if idx in filtered_vocab_set], dtype=torch.long
                    )
                )

        # If it's the first token, force a numeric first token
        else:
            # All sequences can start with any number (0-9) but NOT a '.'
            filtered_vocab_set = set(self.numeric_token_ids.values())

            for _ in range(logits.shape[0]):
                allowed_indices.append(
                    torch.tensor(
                        [idx for idx in range(logits.shape[-1]) if idx in filtered_vocab_set], dtype=torch.long
                    )
                )

        token_ids = torch.empty(logits.shape[0], dtype=torch.long).to(logits.device)
        probs = torch.empty(logits.shape[0], dtype=torch.float32).to(logits.device)

        # For each sequence, apply the filtering to the output logits and select the highest logit
        for i in range(logits.shape[0]):
            filtered_logits = logits[i, allowed_indices[i]]
            max_probs, max_indices = torch.max(filtered_logits, dim=-1)
            token_ids[i] = allowed_indices[i][max_indices]
            probs[i] = max_probs

        return token_ids.unsqueeze(1), probs.unsqueeze(1)
