"""
Compare heuristic vs RF(bracelet pseudo-trained) on local JSON ECG files.

Files used (project root):
  - ЭКГ в покое.json
  - ЭКГ шум.json
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pulsem.data_loader import ECGDataLoader
from ecg_artifact_filter import ECGArtifactFilter, ECGArtifactFilterConfig
from ecg_plotting import plot_ecg_with_artifacts


def summarize(name: str, res: dict) -> str:
    mask = res["artifact_mask"].astype(bool)
    frac = mask.mean() if mask.size else 0.0
    segs = res["artifact_segments"]
    n_segs = len(segs)
    probs = res["artifact_probs"].astype(float)
    mean_prob = float(probs.mean()) if probs.size else 0.0
    return (
        f"{name}: artifact_fraction={frac:.3f}, "
        f"n_segments={n_segs}, mean_artifact_prob={mean_prob:.3f}"
    )


def main() -> None:
    loader = ECGDataLoader()

    files = [
        ("ЭКГ в покое.json", "rest_ecg"),
        ("ЭКГ шум.json", "noisy_ecg"),
    ]

    out_dir = PROJECT_ROOT / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    heur = ECGArtifactFilter(config=ECGArtifactFilterConfig(fs=125))

    rf_model_path = PROJECT_ROOT / "artifacts" / "ecg_rf_artifact_model_bracelet_pseudo.joblib"
    # Postprocessing tuned to avoid segment “chatter”:
    # - smooth probs (~0.8s)
    # - require >=1.0s artifact duration
    # - allow small gaps (~0.25s)
    rf = ECGArtifactFilter.from_joblib(
        str(rf_model_path),
        config=ECGArtifactFilterConfig(
            fs=125,
            prob_threshold=0.6,
            prob_smooth_window=101,
            min_artifact_duration=125,
            gap_tolerance=30,
            use_baseline_deviation_rule=True,
            baseline_window=125,
            baseline_std_k=4.0,
        ),
    )

    for filename, tag in files:
        path = PROJECT_ROOT / filename
        signal, fs = loader.load_json(str(path))

        res_h = heur.infer(signal)
        fig_h = plot_ecg_with_artifacts(
            signal,
            fs=float(fs),
            artifact_mask=res_h["artifact_mask"],
            normal_mask=res_h["normal_mask"],
            title=f"Heuristic {tag}",
        )
        out_h = out_dir / f"compare2_{tag}_heuristic.png"
        fig_h.savefig(out_h, dpi=160)
        print(f"Saved: {out_h}")
        print("  " + summarize(f"Heuristic {tag}", res_h))

        res_r = rf.infer(signal)
        fig_r = plot_ecg_with_artifacts(
            signal,
            fs=float(fs),
            artifact_mask=res_r["artifact_mask"],
            normal_mask=res_r["normal_mask"],
            title=f"RF(bracelet) {tag}",
        )
        out_r = out_dir / f"compare2_{tag}_rf_bracelet.png"
        fig_r.savefig(out_r, dpi=160)
        print(f"Saved: {out_r}")
        print("  " + summarize(f"RF(bracelet) {tag}", res_r))


if __name__ == "__main__":
    main()

