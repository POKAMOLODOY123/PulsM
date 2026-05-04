"""
REST API для инференса PackingTS: один эндпоинт POST /predict.
Запуск из корня проекта: uvicorn api.main:app --host 0.0.0.0 --port 8000
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from api.inference import load_config_and_model, run_inference

app = FastAPI(title="PackingTS Inference API", version="0.1.0")

# Загрузка при старте
_config, _model = None, None


def get_config_and_model():
    global _config, _model
    if _model is None:
        try:
            _config, _model = load_config_and_model()
        except Exception as e:
            raise RuntimeError(f"Failed to load config/model: {e}") from e
    return _config, _model


class PredictRequest(BaseModel):
    """Либо file_id (читать файл из data/raw/series), либо speed (массив)."""
    file_id: str | None = None
    speed: list[float] | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(body: PredictRequest):
    """
    Предсказание числа упаковок и границ сегментов.
    - file_id: загружает серию из data/raw/series/{file_id}.txt (колонка Speed_2).
    - speed: массив значений скорости (например с датчика).
    """
    if body.file_id is None and body.speed is None:
        raise HTTPException(status_code=400, detail="Provide either file_id or speed")
    if body.file_id is not None and body.speed is not None:
        raise HTTPException(status_code=400, detail="Provide only one of file_id or speed")

    config, model = get_config_and_model()
    series_dir = Path(config['paths']['series_dir'])
    target_col = config['data']['target_col']

    if body.file_id is not None:
        filepath = series_dir / f"{body.file_id.strip()}.txt"
        if not filepath.exists():
            raise HTTPException(status_code=404, detail=f"Series file not found: {body.file_id}.txt")
        try:
            df = pd.read_csv(filepath, sep=';', parse_dates=['Time'])
            df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
            speed = df[target_col].fillna(0).values
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read series file: {e}") from e
    else:
        speed = np.array(body.speed, dtype=float)

    if len(speed) == 0:
        raise HTTPException(status_code=400, detail="Empty speed array")

    try:
        result = run_inference(speed, config, model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return result
