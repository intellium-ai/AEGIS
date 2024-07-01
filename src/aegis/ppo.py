import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import AutoModelForCausalLMWithValueHead
from trl import PPOTrainer, PPOConfig
from peft.tuners.prefix_tuning import PrefixTuningConfig
from peft.tuners.lora import LoraConfig
from peft.mapping import get_peft_model

# Load the base model and tokenizer
model_name = (
    "bigscience/bloomz-560m"  # You can change this to any other compatible model
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    model_name,
    peft_config=lora_config,
)
