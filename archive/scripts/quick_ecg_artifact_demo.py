"""
Quick demo: run ECGArtifactFilter + Novaroll-like plotting on prepared MIT-BIH segments.

Prerequisite:
  python scripts/download_datasets.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ecg_artifact_filter import ECGArtifactFilter, ECGArtifactFilterConfig
from ecg_plotting import plot_ecg_with_artifacts


def _pick_first_index(labels: np.ndarray, target: int) -> int:
    idxs = np.where(labels == target)[0]
    if idxs.size == 0:
        raise RuntimeError(f"No samples found for label={target}")
    return int(idxs[0])


def main() -> None:
    dataset_path = PROJECT_ROOT / "data" / "processed" / "mit_bih_segments.npz"
    out_dir = PROJECT_ROOT / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(dataset_path)
    signals = data["signals"]  # (N, 2000)
    labels = data["labels"]  # 0,1,2

    # В этом проекте MIT-BIH ресемплится к длине 2000, но частота для синтетики
    # и дальнейших пайплайнов обычно 200 Гц (см. pulsem/datasets/mit_bih.py)
    fs = 200.0

    config = ECGArtifactFilterConfig(
        fs=125,  # для этого демо не критично (маска в отсчётах)
        prob_threshold=0.5,
        min_artifact_duration=15,
        gap_tolerance=8,
    )
    filt = ECGArtifactFilter(config=config)

    class_names = {0: "Normal", 1: "Mechanical_Artifact", 2: "Heart_Issue"}

    for cls in (0, 1, 2):
        i = _pick_first_index(labels, cls)
        ecg = np.asarray(signals[i], dtype=float).reshape(-1)
        res = filt.infer(ecg)

        fig = plot_ecg_with_artifacts(
            ecg,
            fs=fs,
            artifact_mask=res["artifact_mask"],
            normal_mask=res["normal_mask"],
            title=f"Demo: {class_names[cls]} (idx={i})",
        )
        out_path = out_dir / f"demo_ecg_{class_names[cls].lower()}.png"
        fig.savefig(out_path, dpi=160)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

