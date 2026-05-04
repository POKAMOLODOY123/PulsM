"""
Plot Novaroll-like marking for all bracelet JSON files using RF(bracelet) + postprocessing.

Input:
  data/saveliy.dubovik@gmail.com/*.json

Output:
  artifacts/rf_bracelet_<stem>__partNN.png
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pulsem.data_loader import ECGDataLoader
from ecg_artifact_filter import ECGArtifactFilter, ECGArtifactFilterConfig
from ecg_plotting import plot_ecg_with_artifacts


def chunk_ranges(n: int, chunk_size: int) -> Tuple[Tuple[int, int], ...]:
    if n <= 0 or chunk_size <= 0:
        return tuple()
    ranges = []
    for s in range(0, n, chunk_size):
        e = min(n, s + chunk_size)
        ranges.append((s, e))
    return tuple(ranges)


def main() -> None:
    data_dir = PROJECT_ROOT / "data" / "saveliy.dubovik@gmail.com"
    out_dir = PROJECT_ROOT / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = ECGDataLoader()

    rf_model_path = PROJECT_ROOT / "artifacts" / "ecg_rf_artifact_model_bracelet_pseudo.joblib"
    rf = ECGArtifactFilter.from_joblib(
        str(rf_model_path),
        config=ECGArtifactFilterConfig(
            fs=125,
            prob_threshold=0.6,
            prob_smooth_window=101,
            min_artifact_duration=125,
            gap_tolerance=30,
        ),
    )

    seconds_per_plot = 120  # 2 minutes per image
    max_parts_per_file = 12  # cap to avoid dozens of images for very long recordings

    for p in sorted(data_dir.glob("*.json")):
        sig, fs = loader.load_json(str(p), default_fs=125)
        if sig.size == 0 or fs != 125:
            print(f"Skip {p.name}: empty or fs={fs}")
            continue

        res = rf.infer(sig)
        artifact_mask = np.asarray(res["artifact_mask"], dtype=bool)
        normal_mask = np.asarray(res["normal_mask"], dtype=bool)

        chunk_size = int(seconds_per_plot * fs)
        parts = chunk_ranges(sig.size, chunk_size)
        if len(parts) > max_parts_per_file:
            parts = parts[:max_parts_per_file]

        for i, (s, e) in enumerate(parts, start=1):
            fig = plot_ecg_with_artifacts(
                sig[s:e],
                fs=float(fs),
                artifact_mask=artifact_mask[s:e],
                normal_mask=normal_mask[s:e],
                title=f"RF(bracelet) {p.name} part {i} ({s/fs:.0f}-{e/fs:.0f}s)",
            )
            out_path = out_dir / f"rf_bracelet_{p.stem}__part{i:02d}.png"
            fig.savefig(out_path, dpi=140)
            plt.close(fig)
            print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

