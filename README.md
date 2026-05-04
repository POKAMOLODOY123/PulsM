# PulseM: детекция артефактов ЭКГ

Проект определяет некачественные участки ЭКГ (шум, трение, потеря контакта датчика) и возвращает интервалы:
- `artifact` — участок лучше исключить из расчёта метрик
- `normal` — участок нормального качества

---

## Как работает алгоритм

1. На вход: 1-канальный ЭКГ сигнал (`values`) и частота (`samplingRateHz`).
2. Нормализация сигнала.
3. Построение признаков на нескольких окнах времени.
4. Модель Random Forest оценивает вероятность артефакта по каждой точке.
5. Постобработка (сглаживание, порог, склейка коротких разрывов, минимальная длина сегмента).
6. На выходе: интервалы `artifact` и `normal`.

---

## Функциональность

- API для анализа ЭКГ из JSON тела запроса
- API для анализа загруженного JSON файла
- Возврат интервалов в сэмплах и секундах
- Swagger UI для тестирования
- Docker-образ для запуска в контейнере

---

## Структура

- `service_api/main.py` — FastAPI микросервис
- `ecg_artifact_filter.py` — основная логика фильтрации/детекции
- `artifacts/ecg_rf_artifact_model_labelstudio.joblib` — обученная RF модель
- `requirements-api.txt` — зависимости для API
- `Dockerfile.api` — контейнер API

Архив исследовательского/legacy кода:
- `archive/scripts`
- `archive/root_legacy`

---

## Быстрый запуск

```bash
pip install -r requirements-api.txt
uvicorn service_api.main:app --host 0.0.0.0 --port 8000
```

Проверка:
- Swagger: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- Health: `http://localhost:8000/health`

---

## Запуск в Docker

```bash
docker build -f Dockerfile.api -t pulsem-ecg-api .
docker run --rm -p 8000:8000 pulsem-ecg-api
```

Опционально можно передать путь к модели через env:

```bash
docker run --rm -p 8000:8000 -e MODEL_PATH=artifacts/ecg_rf_artifact_model_labelstudio.joblib pulsem-ecg-api
```

---

## API

### `GET /health`
Проверка доступности сервиса.

Пример ответа:
```json
{"status":"ok"}
```

### `POST /v1/analyze`
Анализ ЭКГ из JSON тела запроса.

Пример запроса:
```json
{
  "values": [933.1, 932.8, 934.0, 931.7],
  "samplingRateHz": 125
}
```

### `POST /v1/analyze-file`
Анализ ЭКГ из загруженного JSON-файла формата:
```json
{
  "values": [...],
  "samplingRateHz": 125
}
```

---

## Формат ответа

Сервис возвращает:
- `summary` — общая статистика
- `artifact_intervals` — интервалы артефактов
- `normal_intervals` — интервалы нормальных участков

Пример фрагмента ответа:
```json
{
  "summary": {
    "artifact_fraction": 0.104,
    "artifact_segment_count": 5,
    "normal_segment_count": 6
  },
  "artifact_intervals": [
    {
      "start_sample": 7500,
      "end_sample": 8700,
      "start_sec": 60.0,
      "end_sec": 69.6,
      "duration_sec": 9.608
    }
  ]
}
```

---

## Примечание

Если файл модели не найден, сервис переключится в эвристический режим (без RF), но для рабочего качества рекомендуется запускать с обученной моделью в `artifacts/`.
