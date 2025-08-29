from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import torch

# Load dataset
dataset = load_dataset("json", data_files="soil_moisture_chat.jsonl", split="train")

# Model and tokenizer
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    device_map="auto"
)

# LoRA config
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # common in LLaMA/Mistral
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

# Preprocess
def preprocess(example):
    prompt = f"Instruction: {example['instruction']}\nResponse:"
    inputs = tokenizer(prompt, truncation=True, padding="max_length", max_length=256)
    labels = tokenizer(example["response"], truncation=True, padding="max_length", max_length=256)
    inputs["labels"] = labels["input_ids"]
    return inputs

tokenized = dataset.map(preprocess, batched=True)

# Training
args = TrainingArguments(
    output_dir="./outputs",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=10,
    max_steps=100,  # keep small for now
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="steps",
    save_steps=50,
)

from transformers import Trainer
trainer = Trainer(
    model=model,
    train_dataset=tokenized,
    args=args,
    tokenizer=tokenizer
)

trainer.train()

# Save adapter
model.save_pretrained("./soilmoisture-lora")
    