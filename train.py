import pandas as pd
from datasets import Dataset
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
import evaluate

# Загружаем данные
df = pd.read_csv("manual_check_dataset.csv", engine='python', on_bad_lines='skip')

# Объединяем метки: если checked_label есть — берем её, иначе label_name
df["final_label"] = df["checked_label"].fillna(df["label_name"])

# Убираем строки без метки (если есть)
df = df.dropna(subset=["final_label"])

# Тексты к строкам, пропуски -> пустые строки
df["text"] = df["text"].fillna("").astype(str)

# Оставляем нужные колонки
df = df[["text", "final_label"]].rename(columns={"final_label": "label"})

# Создаём словарь label->id
label_list = sorted(df["label"].unique())
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label2id.items()}

# Маппим метки в числа
df["label"] = df["label"].map(label2id)

# Создаём Dataset
dataset = Dataset.from_pandas(df)

# Загружаем токенизатор и модель
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id,
)

# Предобработка (токенизация без паддинга)
def preprocess_function(examples):
    texts = [str(x) if x is not None else "" for x in examples["text"]]
    # padding=False, токенизатор не паддит, чтобы Trainer не получил тензоры разной длины без DataCollator
    return tokenizer(texts, truncation=True, padding=False, max_length=128)

dataset = dataset.map(preprocess_function, batched=True)

# Убираем старые колонки
dataset = dataset.remove_columns(["text", "__index_level_0__"])

# Устанавливаем формат для PyTorch
dataset.set_format("torch")

# DataCollator с динамическим паддингом
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Метрика accuracy
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)

# Параметры обучения
training_args = TrainingArguments(
    output_dir="./best_model",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

# Trainer с data_collator
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset,
    tokenizer=tokenizer,  # передаём токенизатор
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Запускаем обучение
trainer.train()
trainer.save_model("./best_model")
