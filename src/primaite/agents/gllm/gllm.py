import torch
from typing import List, Tuple, Dict
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
from torch_geometric.nn import GATConv, LayerNorm, global_add_pool
from peft.tuners.lora import LoraConfig
from primaite.agents.git.git_policy import LLM


class GLLM(torch.nn.Module):
    default_peft_config = LoraConfig(task_type="CAUSAL_LM", r=8, lora_dropout=0.1)

    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolLM-1.7B-Instruct",
        ge_device: str = "cuda:0",
        ge_learning_rate: float = 0.005,
        llm_learning_rate: float = 0.005,
        peft_config: LoraConfig = default_peft_config,
    ):
        self.ge_device = ge_device
        self.model_name = model_name
        self.ge_learning_rate = ge_learning_rate
        self.llm_learning_rate = llm_learning_rate

        # Initialise peft_config
        self.peft_config = peft_config

        self.llm = LLM(model_name=self.model_name, peft_config=self.peft_config)
