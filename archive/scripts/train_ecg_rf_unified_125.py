"""
Train a unified per-sample RandomForest artifact detector on 125 Hz JSON ECG files
from both:
  - data/saveliy.dubovik@gmail.com
  - data/German

Labeling strategy (pseudo labels, HARD zones only):
  m1 = heuristic_mask_125Hz
  m2 = rf_json_mask_125Hz

  hard_positive: m1 == 1 and m2 == 1
  hard_negative: m1 == 0 and m2 == 0
  ambiguous:     m1 != m2  (these points are dropped from training)

Output:
  artifacts/ecg_rf_artifact_model_125_unified.joblib
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
from ecg_artifact_filter import (
    ECGArtifactFilter,
    ECGArtifactFilterConfig,
    _build_ecg_features,
)


def downsample_indices(n: int, max_points: int, rng: np.random.Generator) -> np.ndarray:
    if n <= max_points:
        return np.arange(n, dtype=np.int64)
    return rng.choice(n, size=max_points, replace=False).astype(np.int64)


def balanced_sample_indices(y: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    """
    Downsample to roughly balanced classes (0/1) up to max_points.
    """
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    if idx0.size == 0 or idx1.size == 0:
        # fallback: just random subset
        return downsample_indices(y.size, max_points, rng)

    n_each = min(idx0.size, idx1.size, max_points // 2 if max_points >= 2 else 1)
    pick0 = rng.choice(idx0, size=n_each, replace=False)
    pick1 = rng.choice(idx1, size=n_each, replace=False)
    idx = np.concatenate([pick0, pick1])
    rng.shuffle(idx)
    return idx.astype(np.int64)


def main() -> None:
    data_dirs = [
        PROJECT_ROOT / "data" / "saveliy.dubovik@gmail.com",
        PROJECT_ROOT / "data" / "German",
    ]
    # Also include selected root-level JSON files under ./data (e.g. ad-hoc recordings)
    extra_files = [
        PROJECT_ROOT / "data" / "2026-03-31_17_07_37.json",
    ]
    out_path = PROJECT_ROOT / "artifacts" / "ecg_rf_artifact_model_125_unified.joblib"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    loader = ECGDataLoader()

    # We'll normalize the signal once in this script before both teacher masks + feature extraction.
    heur = ECGArtifactFilter(config=ECGArtifactFilterConfig(fs=125, normalize_input=False))
    rf_json_path = PROJECT_ROOT / "artifacts" / "ecg_rf_artifact_model_json.joblib"
    rf_json = ECGArtifactFilter.from_joblib(
        str(rf_json_path),
        # Teacher RF was trained on old feature set: no variance, window_sizes without 188.
        config=ECGArtifactFilterConfig(
            fs=125,
            normalize_input=False,
            include_variance=False,
            window_sizes=(5, 15, 31, 63, 125),
        ),
    )

    cfg = ECGArtifactFilterConfig(normalize_input=True, include_variance=True)
    rng = np.random.default_rng(2026)

    X_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    max_points_per_file = 80_000

    for d in data_dirs:
        files = sorted(d.glob("*.json"))
        print(f"Scanning {d} ({len(files)} files)")
        for p in files:
            signal, fs = loader.load_json(str(p), default_fs=125)
            if signal.size == 0 or fs != 125:
                # skip non-125Hz or empty
                continue

            sig = signal.astype(float)
            if cfg.normalize_input:
                sig = sig - np.median(sig)
                s = float(np.std(sig))
                if s > 1e-6:
                    sig = sig / s

            m1 = np.asarray(heur.infer(sig)["artifact_mask"], dtype=bool)
            m2 = np.asarray(rf_json.infer(sig)["artifact_mask"], dtype=bool)

            # HARD zones: where both teachers agree.
            hard_mask = (m1 == m2)
            if not hard_mask.any():
                print(f"{p.name}: no hard-agreement zones, skipped")
                continue

            y_full = m1.astype(np.int64)  # or m2; they are equal on hard_mask

            feats = _build_ecg_features(sig, cfg.window_sizes, include_variance=cfg.include_variance)

            # Keep only hard-agreement points before downsampling
            feats_hard = feats[hard_mask]
            y_hard = y_full[hard_mask]

            if feats_hard.shape[0] == 0:
                print(f"{p.name}: zero hard points after filtering, skipped")
                continue

            idx = balanced_sample_indices(y_hard, max_points_per_file, rng)
            X_parts.append(feats_hard[idx])
            y_parts.append(y_hard[idx])

            frac_art = float(y_hard.mean()) if y_hard.size else 0.0
            frac_keep = float(hard_mask.mean())
            print(
                f"{p.name}: n={sig.size}, hard_fraction={frac_keep:.3f}, "
                f"used={idx.size}, hard_artifact_fraction={frac_art:.3f}"
            )

    # Extra root-level files
    for p in extra_files:
        if not p.exists():
            continue
        signal, fs = loader.load_json(str(p), default_fs=125)
        if signal.size == 0 or fs != 125:
            continue
        sig = signal.astype(float)
        if cfg.normalize_input:
            sig = sig - np.median(sig)
            s = float(np.std(sig))
            if s > 1e-6:
                sig = sig / s

        m1 = np.asarray(heur.infer(sig)["artifact_mask"], dtype=bool)
        m2 = np.asarray(rf_json.infer(sig)["artifact_mask"], dtype=bool)
        hard_mask = (m1 == m2)
        if not hard_mask.any():
            print(f"{p.name}: no hard-agreement zones, skipped")
            continue

        y_full = m1.astype(np.int64)
        feats = _build_ecg_features(sig, cfg.window_sizes, include_variance=cfg.include_variance)
        feats_hard = feats[hard_mask]
        y_hard = y_full[hard_mask]
        if feats_hard.shape[0] == 0:
            continue

        idx = balanced_sample_indices(y_hard, max_points_per_file, rng)
        X_parts.append(feats_hard[idx])
        y_parts.append(y_hard[idx])
        frac_art = float(y_hard.mean()) if y_hard.size else 0.0
        frac_keep = float(hard_mask.mean())
        print(
            f"{p.name}: n={sig.size}, hard_fraction={frac_keep:.3f}, "
            f"used={idx.size}, hard_artifact_fraction={frac_art:.3f}"
        )

    if not X_parts:
        raise RuntimeError("No 125 Hz JSON files produced training data for unified RF")

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    print(f"Total unified RF samples: {X.shape[0]}, artifact_fraction={float(y.mean()):.3f}")

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
    print(f"Saved unified 125Hz RF model to {out_path}")


if __name__ == "__main__":
    main()

