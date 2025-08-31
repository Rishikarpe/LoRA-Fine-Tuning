from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import torch

# Load dataset
dataset = load_dataset("json", data_files="soil_moisture_chat.jsonl", split="train")

# Model and tokenizer
model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 🔧 Fix padding issue
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    device_map="auto"
)
# 🔧 Align model config
model.config.pad_token_id = model.config.eos_token_id

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
    # Combine instruction and response for causal language modeling
    full_text = f"Instruction: {example['instruction']}\nResponse: {example['response']}{tokenizer.eos_token}"
    
    # Tokenize the full text
    model_inputs = tokenizer(
        full_text,
        truncation=True,
        padding="max_length",
        max_length=128,  # Increased length128r full instruction + response
        return_tensors="pt"
    )
    
    # Create labels (shifted for causal LM)
    labels = model_inputs["input_ids"].clone()
    
    # Find the position of "Response:" to mask the instruction part
    response_start = full_text.find("Response:")
    if response_start != -1:
        # Tokenize the instruction part to find its length
        instruction_part = full_text[:response_start + len("Response:")]
        instruction_tokens = tokenizer(instruction_part, return_tensors="pt")["input_ids"]
        instruction_length = instruction_tokens.shape[1]
        
        # Mask the instruction part in labels (set to -100)
        labels[:, :instruction_length] = -100
    
    model_inputs["labels"] = labels
    return model_inputs

tokenized = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)

# Training
args = TrainingArguments(
    output_dir="./outputs",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=10,
    max_steps=2000,  # keep small for now
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="steps",
    save_steps=50,
    report_to="none"  # Disable wandb if not configured
)

trainer = Trainer(
    model=model,
    train_dataset=tokenized,
    args=args,
    tokenizer=tokenizer
)

trainer.train()

# Save adapter
model.save_pretrained("./soilmoisture-lora")
