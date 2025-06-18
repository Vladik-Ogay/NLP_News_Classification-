import random
import pandas as pd
from datasets import load_dataset

SEED = 42
random.seed(SEED)

# Классы AG News
label2name = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

# Загрузка
print("Загружаем датасет AG News...")
raw_ds = load_dataset("ag_news", split="train")

# Отбор по 1000 на класс
def select_per_class(dataset, n_per_class=1000, seed=SEED):
    selected = []
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    for i in indices:
        example = dataset[i]
        label = int(example["label"])
        if counts[label] < n_per_class:
            selected.append({
                "text": example["text"],
                "original_label": label,
                "label_name": label2name[label],
                "checked_label": "",   # вручную заполняется потом
                "comment": ""
            })
            counts[label] += 1
        if all(c >= n_per_class for c in counts.values()):
            break
    return selected

subset = select_per_class(raw_ds, n_per_class=1000)

# Сохраняем как CSV
df = pd.DataFrame(subset)
df.to_csv("manual_check_dataset.csv", index=False)
print("✅ Сохранено в manual_check_dataset.csv. Проверьте и исправьте метки.")
