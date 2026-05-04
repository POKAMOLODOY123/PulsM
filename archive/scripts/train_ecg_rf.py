"""
Train a simple RandomForest classifier on MIT-BIH segments
and save it for use in ECGArtifactFilter.

Prerequisite:
  python scripts/download_datasets.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ecg_artifact_filter import ECGArtifactFilterConfig, _build_ecg_features


def main() -> None:
    project_root = PROJECT_ROOT
    dataset_path = project_root / "data" / "processed" / "mit_bih_segments.npz"
    data = np.load(dataset_path)
    signals = data["signals"]  # (N, 2000)
    labels = data["labels"]  # 0: Normal, 1: Mechanical, 2: Heart Issue

    # Binary label: 0 = Normal, 1 = Any artifact (mechanical or heart issue)
    y = (labels != 0).astype(int)

    config = ECGArtifactFilterConfig()

    # Build per-timepoint features then aggregate over each segment
    # to get one feature vector per segment (mean over time).
    feats_list = []
    for seg in signals:
        per_point = _build_ecg_features(seg.astype(float), config.window_sizes)
        feats_list.append(per_point.mean(axis=0))
    X = np.stack(feats_list, axis=0)

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

    model_path = project_root / "artifacts" / "ecg_rf_artifact_model.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_path)
    print(f"Saved RandomForest model to {model_path}")


if __name__ == "__main__":
    main()

