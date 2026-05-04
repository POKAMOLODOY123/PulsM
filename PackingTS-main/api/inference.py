"""
Инференс для веб-API: построение фичей как в обучении, предсказание, постобработка.
"""
import sys
from pathlib import Path
import yaml
import joblib
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.postprocessing import smart_event_counter_v3


def build_features(speed: np.ndarray, window_sizes: list) -> pd.DataFrame:
    """Строит те же фичи, что и create_dataset в data_loader (без разметки)."""
    features = pd.DataFrame({'speed': speed})
    features['speed_lag_50'] = features['speed'].shift(50).fillna(0)
    for w in window_sizes:
        roll = features['speed'].rolling(w, center=True, min_periods=1)
        features[f'roll_mean_{w}'] = roll.mean()
        features[f'roll_max_{w}'] = roll.max()
        features[f'roll_std_{w}'] = roll.std()
        features[f'zero_ratio_{w}'] = (features['speed'] == 0).rolling(w, center=True, min_periods=1).mean()
    return features.fillna(0)


def load_config_and_model(config_path: Path = None, model_path: Path = None):
    """Загружает config и модель из путей по умолчанию относительно ROOT."""
    config_path = config_path or ROOT / "configs" / "config.yaml"
    model_path = model_path or ROOT / "models" / "best_rf_model.joblib"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    model = joblib.load(model_path)
    # Один поток для предсказания — избегаем проблем с multiprocessing в Docker/Windows
    if hasattr(model, 'set_params'):
        model.set_params(n_jobs=1)
    return config, model


def run_inference(
    speed: np.ndarray,
    config: dict,
    model,
    *,
    threshold: float = None,
    min_duration: int = None,
    gap_tolerance: int = None,
) -> dict:
    """
    Предсказание числа упаковок и сегментов по массиву скорости.
    Возвращает {"count": int, "segments": [[start, end], ...]}.
    """
    window_sizes = config['data']['window_sizes']
    X = build_features(speed, window_sizes)
    probs = model.predict_proba(X)[:, 1]

    pp = config.get('postprocessing', {})
    threshold = threshold if threshold is not None else pp.get('inference_threshold', 0.5)
    min_duration = min_duration if min_duration is not None else pp.get('default_min_duration', 100)
    gap_tolerance = gap_tolerance if gap_tolerance is not None else pp.get('default_gap_tolerance', 5)

    n_pred, segments = smart_event_counter_v3(
        probs, speed, threshold=threshold, min_duration=min_duration, gap_tolerance=gap_tolerance
    )
    return {
        "count": n_pred,
        "segments": [[int(s[0]), int(s[1])] for s in segments],
    }
