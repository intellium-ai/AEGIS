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
        self.model: AutoModelForCausalLMWithValueHead = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=self.bnb_config,
            peft_config=self.peft_config,
        )

        self.model.gradient_checkpointing_enable()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, torch_dtype=torch.float16, padding=True, device_map="auto"
        )

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
        attention_mask = torch.cat([torch.ones((batch_size, n_tokens)), attention_mask], dim=1)

        # Expand all sequences to have minimum of close_msg_inputs tokens as slack
        n_tokens = self.close_msg_inputs["inputs_embeds"].shape[1]
        new_pad_embeds = self.pad_token_inputs["inputs_embeds"].repeat(batch_size, n_tokens, 1)
        new_pad_attns = torch.zeros((batch_size, n_tokens))
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
            inputs = {"input_ids": token_ids, "attention_mask": torch.ones(len(token_ids))}

        if not grad:
            with torch.no_grad():
                inputs["inputs_embeds"] = self.model.pretrained_model.get_input_embeddings()(inputs["input_ids"])
        else:
            inputs["inputs_embeds"] = self.model.pretrained_model.get_input_embeddings()(
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
        max_new_tokens: int = 10,
        restrict_output: bool = True,
        last_hidden_state=False,
    ) -> Tuple[List[int], torch.Tensor]:

        next_token_ids = torch.empty(0, dtype=torch.int8)
        next_token_probs = torch.empty(0)
        n_tokens = 0
        stop_generating = False
        per_token_attention = torch.ones((inputs_embeds.shape[0], 1))
        # Generate until max_new_tokens reached
        while not stop_generating:
            if not grad:
                with torch.no_grad():
                    output = self.model.forward(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
            else:
                output = self.model.forward(inputs_embeds=inputs_embeds, attention_mask=attention_mask)

            # Get next logits for each input sequence
            logits = output[1]["logits"][:, -1]

            # Apply restriction on the output logits (or don't)
            if restrict_output:
                if next_token_ids.numel() != 0:
                    last_token_ids = next_token_ids[:, -1].tolist()

                else:
                    last_token_ids = None
                token_ids, probs = self._filter_logits(logits=logits, prev_token_ids=last_token_ids)
            else:
                logits = torch.max(logits, dim=-1)
                token_ids = logits.indices.unsqueeze(1)
                probs = logits.values.unsqueeze(1)

            # Update probs and token ids
            next_token_ids = torch.cat([next_token_ids, token_ids.to("cpu")], dim=1)
            next_token_probs = torch.cat([next_token_probs, probs.to("cpu")], dim=1)

            # Get embeddings of the new token
            new_embeddings = self.get_input_embeddings(token_ids=token_ids, grad=True)["inputs_embeds"]

            # Add the new token to the inputs_embeds and expand the attention mask accordingly
            # TODO: You need to be checking here for EOS token and setting AM to 0 and stopping this sequence generation (manually adding the EOS token ID and AM 0).
            inputs_embeds = torch.cat([inputs_embeds, new_embeddings], dim=1)
            attention_mask = torch.cat([attention_mask, per_token_attention], dim=1)
            n_tokens += 1

            # Check if eos token or max_new_tokens limit reached
            if n_tokens == max_new_tokens:
                stop_generating = True

        # Collect the last hidden states
        if last_hidden_state:
            # Get the output hidden states of only the response
            last_hidden_state = self.model.forward(
                inputs_embeds=inputs_embeds, attention_mask=attention_mask, output_hidden_states=True
            )[1]["hidden_states"][-1][:, -1:, :]
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
