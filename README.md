# Анализ временных рядов — веб-приложение

Streamlit-приложение для анализа данных временных рядов: загрузка CSV, обучение XGBoost, визуализации и статистические тесты (Люнга–Бокса, HAC, Диболда–Мариано).

## Требования

- Python 3.9+
- Зависимости из `requirements.txt`

## Установка

```bash
cd webapp
python -m venv venv
venv\Scripts\activate   # Windows
# или: source venv/bin/activate   # Linux/macOS
pip install -r requirements.txt
```

## Запуск

```bash
streamlit run app.py
```

Откроется браузер по адресу http://localhost:8501.

## Формат данных

Загружаемый CSV должен содержать:

- **order_date** — дата (для осей графиков)
- **test_period** — 0 = обучающая выборка, 1 = тестовая
- Целевая колонка вида `{platform}_{product}_revenue`, например `tta_android_avia_revenue`
- Остальные колонки — признаки для модели

В сайдбаре задаются **Платформа** и **Продукт** (по умолчанию `tta_android`, `avia`).

## Структура репозитория

```
webapp/
├── app.py              # Точка входа Streamlit
├── data_processor.py   # Логика анализа и построение графиков
├── requirements.txt
├── .streamlit/
│   └── config.toml     # Тёмная тема по умолчанию
├── data/               # Примеры данных (опционально)
│   └── example.csv
└── README.md
```

## Лицензия

По усмотрению правообладателя.
