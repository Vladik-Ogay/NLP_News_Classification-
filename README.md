# 📰 News Classification with BERT + Manual Verification

Этот проект — практическая система классификации новостей на основе модели BERT, с возможностью **ручной валидации** и дообучения на размеченных примерах.

## 📌 Описание

Мы используем датасет [AG News](https://huggingface.co/datasets/ag_news) и модель `bert-base-uncased` из Transformers. Проект включает:

- 📥 Загрузку и отбор 1000 новостей по каждому классу (4 класса)
- 🧑‍💻 Веб-интерфейс для ручной валидации меток через Streamlit
- 🔁 Объединение проверенных и оригинальных меток
- 🧠 Обучение модели BERT на итоговом датасете
- 🤖 Предсказания по произвольным текстам с сохранённой моделью

## 🗂️ Структура проекта
news_classification/
├── prepare_for_manual_check.py # Подготовка выборки из AG News
├── manual_check_dataset.csv # CSV для ручной валидации
├── news_labeler.py # Streamlit-интерфейс для разметки
├── merge_manual_labels.py # Объединение оригинальных и проверенных меток
├── train.py # Fine-tuning BERT
├── infer.py # Скрипт для предсказаний
├── best_model/ # Сохранённая обученная модель
└── README.md # Этот файл 🙂
