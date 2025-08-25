from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

model_name = "mistralai/Mistral-7B-v0.1"  # or Instruct variant if public

tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,   # saves VRAM (needs bitsandbytes)
    device_map="auto",
    use_auth_token=True
)
