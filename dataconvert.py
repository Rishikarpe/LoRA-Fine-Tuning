import json
import pandas as pd

df = pd.read_csv("dataset/Daily data of Soil Moisture 2024 Maharashtra Ratnagiri.csv")

qa_pairs = []

for _, row in df.iterrows():
    date = row["Date"]
    moisture = row["Avg_smlvl_at15cm"]
    agency = row["Agency_name"]

    instruction = f"What was the soil moisture on {date} in Ratnagiri?"
    response = f"On {date}, the soil moisture at 15cm depth in Ratnagiri was {moisture} according to {agency}."
    
    qa_pairs.append({"instruction": instruction, "response": response})

# Save to JSONL
with open("soil_moisture_chat.jsonl", "w", encoding="utf-8") as f:
    for entry in qa_pairs:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")