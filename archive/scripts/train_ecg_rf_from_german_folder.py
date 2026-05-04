"""
Train a per-sample RandomForest artifact detector on German JSON ECG files (fs=200 Hz).

Labeling strategy (pseudo labels):
  teacher_mask = OR(heuristic_mask, rf_json_mask_resampled)

For simplicity we keep fs=200 and only reuse the JSON-trained RF as weak teacher
without exact time alignment (it will act mostly as a prior on “difficult” zones).

Input:
  data/German/*.json

Output:
  artifacts/ecg_rf_artifact_model_german_pseudo.joblib
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import List

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
    data_dir = PROJECT_ROOT / "data" / "German"
    out_path = PROJECT_ROOT / "artifacts" / "ecg_rf_artifact_model_german_pseudo.joblib"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    loader = ECGDataLoader()

    # heuristic teacher configured for fs=200
    heur = ECGArtifactFilter(config=ECGArtifactFilterConfig(fs=200))

    # JSON-trained RF used as rough prior (fs mismatch tolerated for now)
    rf_json_path = PROJECT_ROOT / "artifacts" / "ecg_rf_artifact_model_json.joblib"
    rf_json = ECGArtifactFilter.from_joblib(
        str(rf_json_path),
        config=ECGArtifactFilterConfig(fs=125),
    )

    cfg = ECGArtifactFilterConfig()
    rng = np.random.default_rng(123)

    X_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []

    max_points_per_file = 80_000

    all_files = sorted(data_dir.glob("*.json"))
    if not all_files:
        raise RuntimeError(f"No JSON files found in {data_dir}")

    # Берём только половину файлов, чтобы избежать “тяжёлых” кейсов.
    n_files = len(all_files)
    n_keep = max(1, n_files // 2)
    keep_idxs = rng.choice(n_files, size=n_keep, replace=False)
    files = [all_files[i] for i in sorted(keep_idxs)]
    print(f"Using {len(files)}/{n_files} German files for RF training")

    for p in files:
        sig, fs = loader.load_json(str(p), default_fs=200)
        if sig.size == 0 or fs <= 0:
            print(f"Skip {p.name}: empty or fs={fs}")
            continue
        if fs != 200:
            print(f"Skip {p.name}: unexpected fs={fs}")
            continue

        sig_f = sig.astype(float)

        # heuristic teacher at 200 Hz
        m1 = np.asarray(heur.infer(sig_f)["artifact_mask"], dtype=bool)

        # JSON-RF teacher: we just run it on the same raw signal;
        # despite training at 125 Hz it will still highlight noisy regions.
        m2 = np.asarray(rf_json.infer(sig_f)["artifact_mask"], dtype=bool)

        y_full = (m1 | m2).astype(np.int64)
        feats = _build_ecg_features(sig_f, cfg.window_sizes)

        idx = downsample_indices(feats.shape[0], max_points_per_file, rng)
        X_parts.append(feats[idx])
        y_parts.append(y_full[idx])

        print(
            f"{p.name}: n={sig.size}, used={idx.size}, "
            f"teacher_artifact_fraction={float(y_full.mean()):.3f}"
        )

    if not X_parts:
        raise RuntimeError("No German JSON files produced training data")

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    print(f"Total German RF samples: {X.shape[0]}, artifact_fraction={float(y.mean()):.3f}")

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
    print(f"Saved German RF model to {out_path}")


if __name__ == "__main__":
    main()

