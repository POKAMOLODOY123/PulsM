"""
Train a per-sample RandomForest artifact detector on bracelet JSON ECG files.

Labeling strategy (pseudo labels):
  teacher_mask = OR(heuristic_mask, rf_json_mask)

This increases training data without manual annotation.

Input:
  data/saveliy.dubovik@gmail.com/*.json

Output:
  artifacts/ecg_rf_artifact_model_bracelet_pseudo.joblib
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import List, Tuple

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
from ecg_artifact_filter import ECGArtifactFilter, ECGArtifactFilterConfig, _build_ecg_features


def downsample_indices(n: int, max_points: int, rng: np.random.Generator) -> np.ndarray:
    if n <= max_points:
        return np.arange(n, dtype=np.int64)
    return rng.choice(n, size=max_points, replace=False).astype(np.int64)


def main() -> None:
    data_dir = PROJECT_ROOT / "data" / "saveliy.dubovik@gmail.com"
    out_path = PROJECT_ROOT / "artifacts" / "ecg_rf_artifact_model_bracelet_pseudo.joblib"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    loader = ECGDataLoader()

    heur = ECGArtifactFilter(config=ECGArtifactFilterConfig(fs=125))
    rf_json_path = PROJECT_ROOT / "artifacts" / "ecg_rf_artifact_model_json.joblib"
    rf_json = ECGArtifactFilter.from_joblib(str(rf_json_path), config=ECGArtifactFilterConfig(fs=125))

    cfg = ECGArtifactFilterConfig()
    rng = np.random.default_rng(42)

    X_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []

    # Keep training size bounded for speed/memory.
    max_points_per_file = 120_000

    for p in sorted(data_dir.glob("*.json")):
        sig, fs = loader.load_json(str(p), default_fs=125)
        if sig.size == 0 or fs != 125:
            print(f"Skip {p.name}: empty or fs={fs}")
            continue

        # Teachers
        m1 = np.asarray(heur.infer(sig)["artifact_mask"], dtype=bool)
        m2 = np.asarray(rf_json.infer(sig)["artifact_mask"], dtype=bool)
        y_full = (m1 | m2).astype(np.int64)

        feats = _build_ecg_features(sig.astype(float), cfg.window_sizes)

        idx = downsample_indices(feats.shape[0], max_points_per_file, rng)
        X_parts.append(feats[idx])
        y_parts.append(y_full[idx])

        print(
            f"{p.name}: n={sig.size}, used={idx.size}, "
            f"teacher_artifact_fraction={float(y_full.mean()):.3f}"
        )

    if not X_parts:
        raise RuntimeError("No bracelet JSON files produced training data")

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    print(f"Total RF samples: {X.shape[0]}, artifact_fraction={float(y.mean()):.3f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=20,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=["Normal", "Artifact"]))

    joblib.dump(clf, out_path)
    print(f"Saved RF model to {out_path}")


if __name__ == "__main__":
    main()

