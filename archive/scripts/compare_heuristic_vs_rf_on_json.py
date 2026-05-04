"""
Compare heuristic vs RandomForest artifact detection on local JSON ECG files.

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

    # Heuristic filter (no model)
    heur_config = ECGArtifactFilterConfig(
        fs=125,
        prob_threshold=0.5,
        min_artifact_duration=25,
        gap_tolerance=10,
    )
    heur_filter = ECGArtifactFilter(config=heur_config)

    # RF-based filter (trained on local JSON ECG via train_ecg_rf_from_json.py)
    rf_model_path = PROJECT_ROOT / "artifacts" / "ecg_rf_artifact_model_json.joblib"
    rf_config = ECGArtifactFilterConfig(
        fs=125,
        prob_threshold=0.5,
        min_artifact_duration=25,
        gap_tolerance=10,
    )
    rf_filter = ECGArtifactFilter.from_joblib(str(rf_model_path), config=rf_config)

    for filename, tag in files:
        path = PROJECT_ROOT / filename
        signal, fs = loader.load_json(str(path))

        # Heuristic
        res_heur = heur_filter.infer(signal)
        fig_h = plot_ecg_with_artifacts(
            signal,
            fs=float(fs),
            artifact_mask=res_heur["artifact_mask"],
            normal_mask=res_heur["normal_mask"],
            title=f"Heuristic {tag} (fs={fs} Hz)",
        )
        out_h = out_dir / f"compare_{tag}_heuristic.png"
        fig_h.savefig(out_h, dpi=160)
        print(f"Saved: {out_h}")
        print("  " + summarize(f"Heuristic {tag}", res_heur))

        # RandomForest
        res_rf = rf_filter.infer(signal)
        fig_r = plot_ecg_with_artifacts(
            signal,
            fs=float(fs),
            artifact_mask=res_rf["artifact_mask"],
            normal_mask=res_rf["normal_mask"],
            title=f"RF {tag} (fs={fs} Hz)",
        )
        out_r = out_dir / f"compare_{tag}_rf.png"
        fig_r.savefig(out_r, dpi=160)
        print(f"Saved: {out_r}")
        print("  " + summarize(f"RF {tag}", res_rf))


if __name__ == "__main__":
    main()

