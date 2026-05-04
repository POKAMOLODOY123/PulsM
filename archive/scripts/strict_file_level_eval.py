"""
Strict file-level evaluation without leakage.

Pipeline:
1) Split labeled files into train/val/test by whole files.
2) Train RF only on train files.
3) Tune postprocessing params on val files only.
4) Report final metrics on unseen test files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple
from urllib.parse import parse_qs, unquote, urlparse

import joblib
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
        for c in [
            Path.home() / "AppData" / "Local" / "label-studio" / "label-studio" / "media" / "upload" / sub,
            Path.home() / "AppData" / "Local" / "label-studio" / "media" / "upload" / sub,
        ]:
            if c.exists():
                return c.resolve()
    return Path()


def labels_from_task(task: Dict[str, Any], t: np.ndarray) -> np.ndarray:
    y = np.full(t.shape[0], -1, dtype=np.int8)
    anns = task.get("annotations", [])
    if not anns:
        return y
    for r in anns[-1].get("result", []):
        if r.get("type") != "timeserieslabels":
            continue
        v = r.get("value", {})
        labs = v.get("timeserieslabels", [])
        if not labs:
            continue
        s = v.get("start")
        e = v.get("end")
        if s is None or e is None or e <= s:
            continue
        lab = str(labs[0]).strip().lower()
        val = 1 if lab == "red" else 0 if lab == "green" else None
        if val is None:
            continue
        m = (t >= float(s)) & (t < float(e))
        if val == 1:
            y[m] = 1
        else:
            y[(m) & (y != 1)] = 0
    return y


def build_task_record(task: Dict[str, Any], cfg: ECGArtifactFilterConfig) -> Tuple[np.ndarray, np.ndarray, str]:
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
    x = df["ecg_raw"].to_numpy(dtype=float)
    y = labels_from_task(task, t)
    labeled = y >= 0
    if not labeled.any():
        return np.empty((0, 0)), np.empty((0,), dtype=int), f"no_labels:{csv_path.name}"

    x_norm = x - np.median(x)
    s = float(np.std(x_norm))
    if s > 1e-6:
        x_norm = x_norm / s

    feats = _build_ecg_features(
        x_norm, cfg.window_sizes, include_variance=getattr(cfg, "include_variance", True)
    )
    return feats[labeled], y[labeled].astype(int), csv_path.name


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float, int, int, int, int]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1, tp, fp, fn, tn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-json", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-files", type=int, default=3)
    parser.add_argument("--val-files", type=int, default=1)
    parser.add_argument("--save-model", default="artifacts/ecg_rf_artifact_model_strict.joblib")
    args = parser.parse_args()

    tasks = json.loads(Path(args.export_json).read_text(encoding="utf-8"))
    cfg = ECGArtifactFilterConfig()

    records: List[Tuple[np.ndarray, np.ndarray, str]] = []
    for i, t in enumerate(tasks, start=1):
        X, y, name = build_task_record(t, cfg)
        if y.size == 0:
            print(f"[task {i}] skip: {name}")
            continue
        print(f"[task {i}] {name}: labeled={y.size}, red_fraction={float((y==1).mean()):.3f}")
        records.append((X, y, name))

    if len(records) < 4:
        raise RuntimeError("Need at least 4 labeled files for strict train/val/test split")

    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(records))
    rng.shuffle(idx)
    n_train = min(args.train_files, len(records) - 2)
    n_val = min(args.val_files, len(records) - n_train - 1)
    train_idx = sorted(idx[:n_train].tolist())
    val_idx = sorted(idx[n_train : n_train + n_val].tolist())
    test_idx = sorted(idx[n_train + n_val :].tolist())

    print(f"\nTrain: {[records[i][2] for i in train_idx]}")
    print(f"Val:   {[records[i][2] for i in val_idx]}")
    print(f"Test:  {[records[i][2] for i in test_idx]}")

    X_train = np.vstack([records[i][0] for i in train_idx])
    y_train = np.concatenate([records[i][1] for i in train_idx])
    if np.unique(y_train).size < 2:
        raise RuntimeError("Train set has one class only. Change seed or split sizes.")

    clf = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        n_jobs=-1,
        random_state=args.seed,
    )
    clf.fit(X_train, y_train)

    model_path = Path(args.save_model)
    if not model_path.is_absolute():
        model_path = (PROJECT_ROOT / model_path).resolve()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_path)

    # Tune decision threshold on validation set only (feature-level strict evaluation)
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    best = None
    ytv, prv = [], []
    for i in val_idx:
        Xf, yf, _ = records[i]
        ytv.append(yf)
        prv.append(clf.predict_proba(Xf)[:, 1])
    yt_val = np.concatenate(ytv)
    pr_val = np.concatenate(prv)
    for th in thresholds:
        yp = (pr_val >= th).astype(int)
        p, r, f1, tp, fp, fn, tn = metrics(yt_val, yp)
        key = (f1, r, p)
        if best is None or key > best[0]:
            best = (key, (th, p, r, f1, tp, fp, fn, tn))

    assert best is not None
    th, p, r, f1, tp, fp, fn, tn = best[1]
    print("\nBest on VAL:")
    print(f"decision_threshold={th}")
    print(f"Red precision={p:.3f}, recall={r:.3f}, f1={f1:.3f}")

    # Final test with same trained RF (feature-level strict evaluation)
    ytt, ypt = [], []
    for i in test_idx:
        Xf, yf, _ = records[i]
        ytt.append(yf)
        ypt.append((clf.predict_proba(Xf)[:, 1] >= th).astype(int))
    yt = np.concatenate(ytt)
    yp = np.concatenate(ypt)
    p, r, f1, tp, fp, fn, tn = metrics(yt, yp)
    print("\n=== STRICT TEST (UNSEEN FILES) ===")
    print(f"TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"Red precision={p:.3f}, recall={r:.3f}, f1={f1:.3f}")
    print(classification_report(yt, yp, target_names=["Green", "Red"], digits=3))


if __name__ == "__main__":
    main()

