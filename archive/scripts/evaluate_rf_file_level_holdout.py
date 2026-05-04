"""
File-level holdout evaluation for ECG RF on Label Studio export.

Unlike point-wise random split, this script splits by whole tasks/files:
- train on some labeled files
- test on unseen labeled files
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ecg_artifact_filter import ECGArtifactFilterConfig, _build_ecg_features


def resolve_csv_path(csv_url: str) -> Path:
    parsed = urlparse(csv_url)
    q = parse_qs(parsed.query)
    rel = q.get("d", [""])[0]
    if rel:
        rel = unquote(rel).replace("/", "\\").lstrip("\\")
        p = (PROJECT_ROOT / rel).resolve()
        if p.exists():
            return p

    path = unquote(parsed.path or "")
    marker = "/data/upload/"
    if marker in path:
        sub = path.split(marker, 1)[1].replace("/", "\\")
        candidates = [
            Path.home() / "AppData" / "Local" / "label-studio" / "label-studio" / "media" / "upload" / sub,
            Path.home() / "AppData" / "Local" / "label-studio" / "media" / "upload" / sub,
        ]
        for c in candidates:
            if c.exists():
                return c.resolve()
    return Path()


def labels_from_task(task: Dict[str, Any], t: np.ndarray) -> np.ndarray:
    y = np.full(t.shape[0], -1, dtype=np.int8)
    anns = task.get("annotations", [])
    if not anns:
        return y
    results = anns[-1].get("result", [])
    for r in results:
        if r.get("type") != "timeserieslabels":
            continue
        v = r.get("value", {})
        start = v.get("start")
        end = v.get("end")
        labels = v.get("timeserieslabels", [])
        if start is None or end is None or end <= start or not labels:
            continue
        lab = str(labels[0]).strip().lower()
        val = 1 if lab == "red" else 0 if lab == "green" else None
        if val is None:
            continue
        idx = (t >= float(start)) & (t < float(end))
        if val == 1:
            y[idx] = 1
        else:
            y[(idx) & (y != 1)] = 0
    return y


def extract_task_xy(task: Dict[str, Any], cfg: ECGArtifactFilterConfig) -> Tuple[np.ndarray, np.ndarray, str]:
    csv_url = task.get("data", {}).get("csv")
    if not csv_url:
        return np.empty((0, 0)), np.empty((0,), dtype=int), "missing_csv"
    csv_path = resolve_csv_path(str(csv_url))
    if not csv_path.exists():
        return np.empty((0, 0)), np.empty((0,), dtype=int), f"not_found:{csv_url}"

    df = pd.read_csv(csv_path)
    if "time" not in df.columns or "ecg_raw" not in df.columns:
        return np.empty((0, 0)), np.empty((0,), dtype=int), f"bad_csv:{csv_path.name}"

    t = df["time"].to_numpy(dtype=float)
    s = df["ecg_raw"].to_numpy(dtype=float)
    y = labels_from_task(task, t)
    labeled = y >= 0
    if not labeled.any():
        return np.empty((0, 0)), np.empty((0,), dtype=int), f"no_labels:{csv_path.name}"

    x = s - np.median(s)
    sd = float(np.std(x))
    if sd > 1e-6:
        x = x / sd
    feats = _build_ecg_features(
        x, cfg.window_sizes, include_variance=getattr(cfg, "include_variance", True)
    )
    return feats[labeled], y[labeled].astype(int), csv_path.name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-json", required=True)
    parser.add_argument("--test-files", default=2, type=int, help="Number of full files for holdout test")
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    tasks = json.loads(Path(args.export_json).read_text(encoding="utf-8"))
    cfg = ECGArtifactFilterConfig()

    per_file: List[Tuple[np.ndarray, np.ndarray, str]] = []
    for i, t in enumerate(tasks, start=1):
        X, y, name = extract_task_xy(t, cfg)
        if y.size == 0:
            print(f"[task {i}] skip: {name}")
            continue
        pos_frac = float((y == 1).mean())
        print(f"[task {i}] {name}: labeled={y.size}, red_fraction={pos_frac:.3f}")
        per_file.append((X, y, name))

    if len(per_file) < 3:
        raise RuntimeError("Need at least 3 labeled files for file-level holdout")

    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(per_file))
    rng.shuffle(idx)
    n_test = max(1, min(args.test_files, len(per_file) - 1))
    test_idx = sorted(idx[:n_test].tolist())
    train_idx = sorted(idx[n_test:].tolist())

    print(f"\nTrain files: {[per_file[i][2] for i in train_idx]}")
    print(f"Test files:  {[per_file[i][2] for i in test_idx]}")

    X_train = np.vstack([per_file[i][0] for i in train_idx])
    y_train = np.concatenate([per_file[i][1] for i in train_idx])
    X_test = np.vstack([per_file[i][0] for i in test_idx])
    y_test = np.concatenate([per_file[i][1] for i in test_idx])

    if np.unique(y_train).size < 2:
        raise RuntimeError("Train split has only one class; change seed or use more train files")
    if np.unique(y_test).size < 2:
        raise RuntimeError("Test split has only one class; change seed or choose different holdout")

    clf = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        n_jobs=-1,
        random_state=args.seed,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    print("\n=== FILE-LEVEL HOLDOUT RESULTS ===")
    print(f"test_points={y_test.size}")
    print(f"TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(classification_report(y_test, y_pred, target_names=["Green", "Red"], digits=3))


if __name__ == "__main__":
    main()

