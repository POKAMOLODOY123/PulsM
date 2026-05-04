"""
Run trained ECG RF model on all tasks from Label Studio export and save plots.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ecg_artifact_filter import ECGArtifactFilter, ECGArtifactFilterConfig
from ecg_plotting import plot_ecg_with_artifacts


def resolve_csv_path(csv_url: str) -> Path:
    parsed = urlparse(csv_url)
    q = parse_qs(parsed.query)
    rel = q.get("d", [""])[0]
    if rel:
        rel = unquote(rel).replace("/", "\\").lstrip("\\")
        p = (PROJECT_ROOT / rel).resolve()
        if p.exists():
            return p

    # /data/upload/<project>/<file>
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


def safe_name(s: str) -> str:
    out = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-json", required=True, help="Label Studio export JSON path")
    parser.add_argument(
        "--model-path",
        default="artifacts/ecg_rf_artifact_model_labelstudio.joblib",
        help="RF model path",
    )
    parser.add_argument(
        "--out-dir",
        default="artifacts/labelstudio_export_rf_plots",
        help="Output directory for plots",
    )
    args = parser.parse_args()

    export_path = Path(args.export_json)
    tasks = json.loads(export_path.read_text(encoding="utf-8"))
    if not isinstance(tasks, list) or not tasks:
        raise RuntimeError("Export JSON must be a non-empty list of tasks")

    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = (PROJECT_ROOT / model_path).resolve()
    if not model_path.exists():
        raise RuntimeError(f"Model not found: {model_path}")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (PROJECT_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

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

    for i, task in enumerate(tasks, start=1):
        csv_url = task.get("data", {}).get("csv")
        if not csv_url:
            print(f"[task {i}] skip: missing data.csv")
            continue
        csv_path = resolve_csv_path(str(csv_url))
        if not csv_path.exists():
            print(f"[task {i}] skip: file not found -> {csv_url}")
            continue

        df = pd.read_csv(csv_path)
        if "ecg_raw" not in df.columns:
            print(f"[task {i}] skip: no ecg_raw column -> {csv_path.name}")
            continue
        ecg = df["ecg_raw"].to_numpy(dtype=float)

        fs = 125.0
        if "time" in df.columns and len(df) > 1:
            t = df["time"].to_numpy(dtype=float)
            dt = np.diff(t)
            dt = dt[np.isfinite(dt) & (dt > 1e-9)]
            if dt.size:
                fs = float(np.round(1.0 / np.median(dt)))

        res = filt.infer(ecg)
        fig = plot_ecg_with_artifacts(
            ecg,
            fs=fs,
            artifact_mask=res["artifact_mask"],
            normal_mask=res["normal_mask"],
            title=f"RF on LS task {i}: {csv_path.name}",
        )
        out_name = f"task_{i:02d}_{safe_name(csv_path.stem)}_rf.png"
        out_path = out_dir / out_name
        fig.savefig(out_path, dpi=160)
        print(
            f"[task {i}] saved: {out_path} | "
            f"artifact_fraction={float(res['artifact_mask'].mean()):.3f}, "
            f"segments={len(res['artifact_segments'])}"
        )


if __name__ == "__main__":
    main()

