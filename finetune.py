# finetune.py

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

# 1. Load dataset
dataset = load_dataset("json", data_files="soil_moisture_chat.jsonl")
print(dataset["train"][0])   # sanity check

# 2. Load base model
model_name = "mistralai/Mistral-7B-v0.1"   # or smaller if GPU is limited
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto"
)

# 3. Wrap model with LoRA
config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # LoRA usually targets attention layers
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)