import pandas as pd

# Читаем CSV с защитой от "лишних" запятых
df = pd.read_csv("manual_check_dataset.csv", quotechar='"', sep=",", encoding="utf-8", on_bad_lines="skip")

# Проверка нужных колонок
if "checked_label" not in df.columns or "original_label" not in df.columns:
    raise ValueError("❌ В CSV должны быть колонки 'checked_label' и 'original_label'.")

# Выбор ручной разметки, если есть — иначе исходной
df["final_label"] = df["checked_label"].fillna(df["original_label"])

# Безопасное приведение к int и фильтрация
def safe_int(x):
    try:
        return int(x)
    except:
        return -1

df["final_label"] = df["final_label"].apply(safe_int)
df = df[df["final_label"].between(0, 3)]

# Только нужные поля
df = df[["text", "final_label"]]
df = df.rename(columns={"final_label": "label"})

# Сохраняем
df.to_csv("merged_dataset.csv", index=False)
print("✅ merged_dataset.csv создан.")