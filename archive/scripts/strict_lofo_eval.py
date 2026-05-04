"""
Strict leave-one-file-out evaluation for ECG RF.

For each labeled file:
- test = this file
- val = one remaining file with highest Red fraction
- train = all other remaining files

Threshold is tuned on val only, then evaluated on unseen test file.
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


def prf(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float, int, int, int, int]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1, tp, fp, fn, tn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-json", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tasks = json.loads(Path(args.export_json).read_text(encoding="utf-8"))
    cfg = ECGArtifactFilterConfig()
    records: List[Tuple[np.ndarray, np.ndarray, str, float]] = []

    for i, t in enumerate(tasks, start=1):
        csv_url = t.get("data", {}).get("csv")
        if not csv_url:
            continue
        csv = resolve_csv_path(str(csv_url))
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        if "time" not in df.columns or "ecg_raw" not in df.columns:
            continue
        tt = df["time"].to_numpy(dtype=float)
        xx = df["ecg_raw"].to_numpy(dtype=float)
        yy = labels_from_task(t, tt)
        m = yy >= 0
        if not m.any():
            print(f"[task {i}] skip no labels: {csv.name}")
            continue
        x = xx - np.median(xx)
        s = float(np.std(x))
        if s > 1e-6:
            x = x / s
        X = _build_ecg_features(x, cfg.window_sizes, include_variance=getattr(cfg, "include_variance", True))
        y = yy[m].astype(int)
        records.append((X[m], y, csv.name, float((y == 1).mean())))
        print(f"[task {i}] {csv.name}: labeled={y.size}, red_fraction={float((y==1).mean()):.3f}")

    if len(records) < 4:
        raise RuntimeError("Need at least 4 labeled files")

    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6]
    per_fold = []
    all_true, all_pred = [], []

    for test_i in range(len(records)):
        # select val among remaining with highest red fraction (non-zero preferred)
        remain = [j for j in range(len(records)) if j != test_i]
        remain_sorted = sorted(remain, key=lambda j: records[j][3], reverse=True)
        val_i = remain_sorted[0]
        train_i = [j for j in remain if j != val_i]

        X_train = np.vstack([records[j][0] for j in train_i])
        y_train = np.concatenate([records[j][1] for j in train_i])
        if np.unique(y_train).size < 2:
            print(f"[fold test={records[test_i][2]}] skip: train one class")
            continue

        clf = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            n_jobs=-1,
            random_state=args.seed,
        )
        clf.fit(X_train, y_train)

        X_val, y_val, _, _ = records[val_i]
        pr_val = clf.predict_proba(X_val)[:, 1]
        best = None
        for th in thresholds:
            yv = (pr_val >= th).astype(int)
            p, r, f1, *_ = prf(y_val, yv)
            key = (f1, r, p)
            if best is None or key > best[0]:
                best = (key, th)
        th = best[1] if best is not None else 0.5

        X_test, y_test, name, red_frac = records[test_i]
        y_pred = (clf.predict_proba(X_test)[:, 1] >= th).astype(int)
        p, r, f1, tp, fp, fn, tn = prf(y_test, y_pred)
        per_fold.append(
            {
                "file": name,
                "red_fraction": red_frac,
                "threshold": th,
                "red_precision": p,
                "red_recall": r,
                "red_f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "n": int(y_test.size),
            }
        )
        all_true.append(y_test)
        all_pred.append(y_pred)
        print(
            f"[fold test={name}] th={th:.2f} "
            f"P={p:.3f} R={r:.3f} F1={f1:.3f} "
            f"TP={tp} FP={fp} FN={fn} TN={tn}"
        )

    yt = np.concatenate(all_true)
    yp = np.concatenate(all_pred)
    p, r, f1, tp, fp, fn, tn = prf(yt, yp)
    print("\n=== LOFO OVERALL (point-weighted) ===")
    print(f"Red precision={p:.3f}, recall={r:.3f}, f1={f1:.3f}")
    print(f"TP={tp}, FP={fp}, FN={fn}, TN={tn}, n={yt.size}")

    out_csv = PROJECT_ROOT / "artifacts" / "strict_lofo_results.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_fold).to_csv(out_csv, index=False)
    print(f"Saved per-fold results: {out_csv}")


if __name__ == "__main__":
    main()

