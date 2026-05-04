"""
Train a RandomForest artifact detector directly on local JSON ECG files.

For now, uses two files in project root:
  - ЭКГ в покое.json  (no artifacts)
  - ЭКГ шум.json      (artifacts at 60–70s and 90–100s)
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, List, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pulsem.data_loader import ECGDataLoader
from ecg_artifact_filter import ECGArtifactFilterConfig, _build_ecg_features


Intervals = List[Tuple[float, float]]


def intervals_to_mask(n_samples: int, fs: int, intervals: Intervals) -> np.ndarray:
    """
    Convert list of (start_sec, end_sec) intervals to boolean mask of length n_samples.
    """
    mask = np.zeros(n_samples, dtype=bool)
    for start_s, end_s in intervals:
        if end_s <= start_s:
            continue
        start_idx = int(round(start_s * fs))
        end_idx = int(round(end_s * fs))
        start_idx = max(0, min(n_samples, start_idx))
        end_idx = max(0, min(n_samples, end_idx))
        if end_idx > start_idx:
            mask[start_idx:end_idx] = True
    return mask


def main() -> None:
    loader = ECGDataLoader()

    # Simple manual labeling for now.
    file_intervals: Dict[str, Intervals] = {
        "ЭКГ в покое.json": [],
        # noisy ECG: 1:00–1:10 loss of contact, 1:30–1:40 friction
        "ЭКГ шум.json": [
            (60.0, 70.0),
            (90.0, 100.0),
        ],
    }

    all_feats: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    config = ECGArtifactFilterConfig()

    for filename, intervals in file_intervals.items():
        path = PROJECT_ROOT / filename
        signal, fs = loader.load_json(str(path))
        n = signal.shape[0]

        per_point_feats = _build_ecg_features(signal.astype(float), config.window_sizes)
        mask = intervals_to_mask(n, fs, intervals)
        y = mask.astype(int)

        all_feats.append(per_point_feats)
        all_labels.append(y)

        frac_art = float(mask.mean()) if mask.size else 0.0
        print(f"{filename}: samples={n}, fs={fs}, artifact_fraction={frac_art:.3f}")

    X = np.vstack(all_feats)
    y = np.concatenate(all_labels)

    print(f"Total samples for RF: {X.shape[0]}, artifact_fraction={y.mean():.3f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=["Normal", "Artifact"]))

    model_path = PROJECT_ROOT / "artifacts" / "ecg_rf_artifact_model_json.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_path)
    print(f"Saved JSON-trained RandomForest model to {model_path}")


if __name__ == "__main__":
    main()

