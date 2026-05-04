"""
Tune RF postprocessing params with file-level holdout evaluation.

Searches best config for:
- prob_threshold
- prob_smooth_window
- min_artifact_duration
- gap_tolerance
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


def score(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float, int, int, int, int]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1, tp, fp, fn, tn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-json", required=True)
    parser.add_argument("--model-path", default="artifacts/ecg_rf_artifact_model_labelstudio.joblib")
    parser.add_argument("--test-files", default=2, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--fast", action="store_true", help="Use smaller grid for quick tuning")
    args = parser.parse_args()

    tasks = json.loads(Path(args.export_json).read_text(encoding="utf-8"))
    valid: List[Tuple[np.ndarray, np.ndarray, str]] = []
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
        tt = df["time"].to_numpy(dtype=float)
        xx = df["ecg_raw"].to_numpy(dtype=float)
        yy = labels_from_task(t, tt)
        lab = yy >= 0
        if lab.any():
            valid.append((xx, yy, csv_path.name))

    if len(valid) < 3:
        raise RuntimeError("Need at least 3 labeled files")

    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(valid))
    rng.shuffle(idx)
    n_test = max(1, min(args.test_files, len(valid) - 1))
    test_idx = sorted(idx[:n_test].tolist())
    train_idx = sorted(idx[n_test:].tolist())

    # Train set is not used for fitting model (already trained), only for reporting split.
    print(f"Train files: {[valid[i][2] for i in train_idx]}")
    print(f"Test files:  {[valid[i][2] for i in test_idx]}")

    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = (PROJECT_ROOT / model_path).resolve()

    if args.fast:
        thresholds = [0.5, 0.6, 0.7]
        smooth_ws = [61, 101]
        min_durs = [75, 125]
        gaps = [10, 30]
    else:
        thresholds = [0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
        smooth_ws = [31, 61, 101, 151]
        min_durs = [50, 75, 100, 125]
        gaps = [10, 20, 30, 40]

    best = None
    rows = []

    combos = list(itertools.product(thresholds, smooth_ws, min_durs, gaps))
    total = len(combos)
    for k, (th, sw, md, gp) in enumerate(combos, start=1):
        filt = ECGArtifactFilter.from_joblib(
            str(model_path),
            config=ECGArtifactFilterConfig(
                fs=125,
                prob_threshold=th,
                prob_smooth_window=sw,
                min_artifact_duration=md,
                gap_tolerance=gp,
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
        all_t = []
        all_p = []
        for i in test_idx:
            x, y, _ = valid[i]
            m = y >= 0
            yp = filt.infer(x)["artifact_mask"].astype(int)
            all_t.append(y[m].astype(int))
            all_p.append(yp[m])
        yt = np.concatenate(all_t)
        yp = np.concatenate(all_p)
        p, r, f1, tp, fp, fn, tn = score(yt, yp)
        rows.append((th, sw, md, gp, p, r, f1, tp, fp, fn, tn))
        if k % 4 == 0 or k == total:
            print(f"Progress: {k}/{total} combos, current best f1={best[0][0]:.3f}" if best else f"Progress: {k}/{total}")
        key = (f1, r, p)
        if best is None or key > best[0]:
            best = (key, (th, sw, md, gp, p, r, f1, tp, fp, fn, tn))

    assert best is not None
    th, sw, md, gp, p, r, f1, tp, fp, fn, tn = best[1]
    print("\n=== BEST CONFIG ON FILE-LEVEL TEST ===")
    print(
        f"prob_threshold={th}, prob_smooth_window={sw}, "
        f"min_artifact_duration={md}, gap_tolerance={gp}"
    )
    print(f"Red precision={p:.3f}, recall={r:.3f}, f1={f1:.3f}")
    print(f"TP={tp}, FP={fp}, FN={fn}, TN={tn}")

    out = PROJECT_ROOT / "artifacts" / "rf_tuning_file_level.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        rows,
        columns=[
            "prob_threshold",
            "prob_smooth_window",
            "min_artifact_duration",
            "gap_tolerance",
            "red_precision",
            "red_recall",
            "red_f1",
            "tp",
            "fp",
            "fn",
            "tn",
        ],
    ).sort_values(["red_f1", "red_recall", "red_precision"], ascending=False)
    df.to_csv(out, index=False)
    print(f"Saved full tuning table: {out}")


if __name__ == "__main__":
    main()

