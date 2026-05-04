"""
Tune postprocessing params to maximize artifact recall.

Goal: make model catch most artifacts (Red recall), accepting extra false positives.
We search a small grid and pick configuration that maximizes:
  1) min per-file Red recall (only files with Red support > 0)
  2) overall Red recall
  3) overall Red precision

Evaluation uses the provided Label Studio export as ground truth (labeled points only).
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ecg_artifact_filter import ECGArtifactFilter, ECGArtifactFilterConfig, _segments_from_probs


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


def pr_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, int, int, int, int]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return prec, rec, tp, fp, fn, tn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-json", required=True)
    ap.add_argument("--model-path", default="artifacts/ecg_rf_artifact_model_labelstudio.joblib")
    ap.add_argument("--out-csv", default="artifacts/tuning_high_recall.csv")
    args = ap.parse_args()

    tasks = json.loads(Path(args.export_json).read_text(encoding="utf-8"))
    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = (PROJECT_ROOT / model_path).resolve()

    # Load per-task raw series + labeled mask once
    task_data: List[Tuple[np.ndarray, np.ndarray, str]] = []
    for t in tasks:
        csv_url = t.get("data", {}).get("csv")
        if not csv_url:
            continue
        csv_path = resolve_csv_path(str(csv_url))
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        if "time" not in df.columns or "ecg_raw" not in df.columns:
            continue
        tt = df["time"].to_numpy(float)
        xx = df["ecg_raw"].to_numpy(float)
        yy = labels_from_task(t, tt)
        labeled = yy >= 0
        if not labeled.any():
            continue
        task_data.append((xx, yy, csv_path.name))

    if not task_data:
        raise RuntimeError("No labeled tasks found")

    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    smooth_ws = [31, 61, 101]
    min_durs = [50, 75, 100]
    gaps = [10, 20, 30]

    # Compute probabilities once (feature extraction is the expensive part).
    base_filt = ECGArtifactFilter.from_joblib(
        str(model_path),
        config=ECGArtifactFilterConfig(
            fs=125,
            normalize_input=True,
            include_variance=True,
            use_baseline_deviation_rule=True,
            baseline_window=125,
            baseline_std_k=3.5,
            use_plateau_rule=False,
            use_flatline_rule=False,
        ),
    )

    cached: List[Tuple[np.ndarray, np.ndarray, str]] = []
    for x, y, name in task_data:
        probs = base_filt._predict_artifact_proba(x)  # noqa: SLF001 (private ok for tuning)
        cached.append((probs, y, name))

    rows = []
    best = None

    combos = list(itertools.product(thresholds, smooth_ws, min_durs, gaps))
    for k, (th, sw, md, gp) in enumerate(combos, start=1):
        cfg = ECGArtifactFilterConfig(
            fs=125,
            prob_threshold=th,
            prob_smooth_window=sw,
            min_artifact_duration=md,
            gap_tolerance=gp,
            # postprocess options
            use_multi_scale=True,
            multi_scale_min_durations=(50, 125, 188),
        )

        per_file_recalls = []
        all_true, all_pred = [], []
        for probs, y, name in cached:
            m = y >= 0
            yt = y[m].astype(int)
            mask, _ = _segments_from_probs(probs, cfg)
            yp = mask.astype(int)[m]
            # per-file recall only if there is at least one red
            if (yt == 1).any():
                _, rec, *_ = pr_metrics(yt, yp)
                per_file_recalls.append(rec)
            all_true.append(yt)
            all_pred.append(yp)

        y_all = np.concatenate(all_true)
        p_all, r_all, tp, fp, fn, tn = pr_metrics(y_all, np.concatenate(all_pred))
        min_rec = float(min(per_file_recalls)) if per_file_recalls else 0.0

        rows.append(
            {
                "prob_threshold": th,
                "prob_smooth_window": sw,
                "min_artifact_duration": md,
                "gap_tolerance": gp,
                "min_file_red_recall": min_rec,
                "overall_red_precision": p_all,
                "overall_red_recall": r_all,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
        )

        key = (min_rec, r_all, p_all)
        if best is None or key > best[0]:
            best = (key, rows[-1])

        if k % 15 == 0 or k == len(combos):
            print(f"Progress {k}/{len(combos)} best_min_recall={best[1]['min_file_red_recall']:.3f}")

    out = Path(args.out_csv)
    if not out.is_absolute():
        out = (PROJECT_ROOT / out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(
        ["min_file_red_recall", "overall_red_recall", "overall_red_precision"], ascending=False
    ).to_csv(out, index=False)

    assert best is not None
    b = best[1]
    print("\nBEST (high recall):")
    print(
        f"th={b['prob_threshold']} smooth={b['prob_smooth_window']} "
        f"min_dur={b['min_artifact_duration']} gap={b['gap_tolerance']}"
    )
    print(
        f"min_file_recall={b['min_file_red_recall']:.3f} "
        f"overall_recall={b['overall_red_recall']:.3f} "
        f"overall_precision={b['overall_red_precision']:.3f}"
    )
    print(f"Saved table: {out}")


if __name__ == "__main__":
    main()

