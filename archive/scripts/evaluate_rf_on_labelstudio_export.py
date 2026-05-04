"""
Evaluate trained RF model against Label Studio annotations.

Computes point-wise metrics on labeled samples only:
- precision / recall / F1 for Red(artifact)
- confusion matrix
- per-task and overall reports
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ecg_artifact_filter import ECGArtifactFilter, ECGArtifactFilterConfig


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
    y = np.full(t.shape[0], -1, dtype=np.int8)  # -1 unlabeled, 0 green, 1 red
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-json", required=True)
    parser.add_argument("--model-path", default="artifacts/ecg_rf_artifact_model_labelstudio.joblib")
    args = parser.parse_args()

    tasks = json.loads(Path(args.export_json).read_text(encoding="utf-8"))
    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = (PROJECT_ROOT / model_path).resolve()

    filt = ECGArtifactFilter.from_joblib(
        str(model_path),
        config=ECGArtifactFilterConfig(
            fs=125,
            prob_threshold=0.6,
            prob_smooth_window=101,
            min_artifact_duration=75,
            gap_tolerance=20,
            normalize_input=True,
            include_variance=True,
            use_baseline_deviation_rule=True,
            baseline_window=125,
            baseline_std_k=3.5,
            use_plateau_rule=False,
            use_flatline_rule=False,
            use_multi_scale=True,
            multi_scale_min_durations=(50, 125, 188),
        ),
    )

    all_true: List[np.ndarray] = []
    all_pred: List[np.ndarray] = []

    for i, task in enumerate(tasks, start=1):
        csv_url = task.get("data", {}).get("csv")
        if not csv_url:
            print(f"[task {i}] skip: no data.csv")
            continue
        csv_path = resolve_csv_path(str(csv_url))
        if not csv_path.exists():
            print(f"[task {i}] skip: csv not found")
            continue

        df = pd.read_csv(csv_path)
        if "time" not in df.columns or "ecg_raw" not in df.columns:
            print(f"[task {i}] skip: missing required columns")
            continue

        t = df["time"].to_numpy(dtype=float)
        ecg = df["ecg_raw"].to_numpy(dtype=float)
        y_true = labels_from_task(task, t)
        labeled = y_true >= 0
        if not labeled.any():
            print(f"[task {i}] skip: no labeled points")
            continue

        pred = filt.infer(ecg)["artifact_mask"].astype(int)
        yt = y_true[labeled].astype(int)
        yp = pred[labeled]
        all_true.append(yt)
        all_pred.append(yp)

        cm = confusion_matrix(yt, yp, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        print(
            f"[task {i}] {csv_path.name}: labeled={yt.size}, "
            f"Red_precision={prec:.3f}, Red_recall={rec:.3f}, Red_f1={f1:.3f}, "
            f"TP={tp}, FP={fp}, FN={fn}, TN={tn}"
        )

    if not all_true:
        raise RuntimeError("No labeled tasks found for evaluation")

    y_true_all = np.concatenate(all_true)
    y_pred_all = np.concatenate(all_pred)
    cm = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    print("\n=== OVERALL ===")
    print(f"labeled_points={y_true_all.size}")
    print(f"TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(classification_report(y_true_all, y_pred_all, target_names=["Green", "Red"], digits=3))


if __name__ == "__main__":
    main()

