"""
Plot high-recall RF artifact detection on selected JSON files/folders.
Saves plots into artifacts/high_recall_plots.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pulsem.data_loader import ECGDataLoader
from ecg_artifact_filter import ECGArtifactFilter, ECGArtifactFilterConfig
from ecg_plotting import plot_ecg_with_artifacts


def iter_json_files(paths: Iterable[Path]) -> List[Path]:
    out: List[Path] = []
    for p in paths:
        if p.is_dir():
            out.extend(sorted(p.glob("*.json")))
        elif p.is_file() and p.suffix.lower() == ".json":
            out.append(p)
    # unique while preserving order
    seen = set()
    uniq = []
    for p in out:
        rp = str(p.resolve())
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    return uniq


def safe_name(s: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in ("-", "_", ".")) else "_" for ch in s)


def main() -> None:
    model_path = PROJECT_ROOT / "artifacts" / "ecg_rf_artifact_model_labelstudio.joblib"
    filt = ECGArtifactFilter.from_joblib(
        str(model_path),
        config=ECGArtifactFilterConfig(
            fs=125,
            prob_threshold=0.3,
            prob_smooth_window=101,
            min_artifact_duration=50,
            gap_tolerance=10,
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

    targets = [
        PROJECT_ROOT / "data" / "GermanNew",
        PROJECT_ROOT / "data" / "gimaratovich@gmail.com",
        PROJECT_ROOT / "data" / "2026-03-31_17_07_37.json",
        PROJECT_ROOT / "data" / "2026-04-11_14_55_31.json",
    ]
    files = iter_json_files(targets)
    if not files:
        raise RuntimeError("No JSON files found in selected targets")

    out_dir = PROJECT_ROOT / "artifacts" / "high_recall_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = ECGDataLoader()
    for p in files:
        try:
            sig, fs = loader.load_json(str(p), default_fs=125)
        except Exception as e:
            print(f"skip invalid json: {p} ({e})")
            continue
        if sig.size == 0:
            print(f"skip empty: {p}")
            continue
        res = filt.infer(sig)
        frac = float(res["artifact_mask"].mean())
        segs = len(res["artifact_segments"])
        fig = plot_ecg_with_artifacts(
            sig,
            fs=float(fs),
            artifact_mask=res["artifact_mask"],
            normal_mask=res["normal_mask"],
            title=f"High-recall RF: {p.name}",
        )
        out_path = out_dir / f"{safe_name(p.stem)}_high_recall.png"
        fig.savefig(out_path, dpi=160)
        print(f"{p.name}: artifact_fraction={frac:.3f}, segments={segs} -> {out_path}")


if __name__ == "__main__":
    main()

