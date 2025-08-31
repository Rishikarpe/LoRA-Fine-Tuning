from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from peft import PeftModel

base_model = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(base_model)

# Quantization config
bnb_config = BitsAndBytesConfig(load_in_4bit=True)

# Load base model in 4-bit
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    quantization_config=bnb_config
)

# Load LoRA adapter
model = PeftModel.from_pretrained(model, "./soilmoisture-lora")

# Define pipeline with generation settings
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto"
)

# Interactive loop
while True:
    instruction = input("Ask me anything? (or type 'quit' to exit) ")
    if instruction.lower() == "quit":
        break

    out = pipe(
        f"Instruction: {instruction}\nResponse:",
        max_new_tokens=150,       # how long the answer can be
        temperature=0.2,          # lower = less randomness
        top_p=0.9,                # nucleus sampling (limits to top 90% probs)
        repetition_penalty=1.2,   # discourage repeating phrases
        do_sample=True            # required for temp/top_p to work
    )

    print("\n" + out[0]['generated_text'] + "\n")
