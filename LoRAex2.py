from peft import LoraConfig, get_peft_model

# Define LoRA config
lora_config = LoraConfig(
    r=16,              # rank (size of trainable matrices)
    lora_alpha=32,     # scaling
    target_modules=["q_proj", "v_proj"],  # which parts of model to adapt
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"   # type of model (chatbot, etc.)
)

# Wrap base model with PEFT
model = get_peft_model(model, lora_config)
