import pandas as pd
import json

df = pd.read_csv("manual_check_dataset.csv")

# Используем checked_label, если указана. Иначе fallback на original_label
cleaned = []
for _, row in df.iterrows():
    label = row["checked_label"]
    if pd.isna(label) or label == "":
        label = row["original_label"]
    cleaned.append({"text": row["text"], "label": int(label)})

# Сохраняем
with open("manual_checked_data.json", "w", encoding="utf-8") as f:
    json.dump(cleaned, f, ensure_ascii=False, indent=2)

print("✅ Сохранено как manual_checked_data.json для обучения модели.")
