from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

# Указываем путь к локальной модели
model_path = "./best_model"

# Загружаем модель и токенизатор как локальные файлы
tokenizer = BertTokenizer.from_pretrained(model_path, local_files_only=True)
model = BertForSequenceClassification.from_pretrained(model_path, local_files_only=True)

# Создаём pipeline
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Пример новости для предсказания
text = "NASA launches a new mission to Mars next week."

# Предсказываем
prediction = classifier(text)
print(prediction)
