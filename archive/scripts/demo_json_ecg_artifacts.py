"""
Demo: run ECGArtifactFilter + plotting on local JSON ECG files.

Uses files in project root:
  - ЭКГ в покое.json  (clean ECG)
  - ЭКГ шум.json      (noisy ECG)
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


def main() -> None:
    loader = ECGDataLoader()

    files = [
        ("ЭКГ в покое.json", "rest_ecg"),
        ("ЭКГ шум.json", "noisy_ecg"),
    ]

    out_dir = PROJECT_ROOT / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    config = ECGArtifactFilterConfig(
        fs=125,
        prob_threshold=0.5,
        min_artifact_duration=25,
        gap_tolerance=10,
    )
    filt = ECGArtifactFilter(config=config)

    for filename, tag in files:
        path = PROJECT_ROOT / filename
        signal, fs = loader.load_json(str(path))

        res = filt.infer(signal)

        fig = plot_ecg_with_artifacts(
            signal,
            fs=float(fs),
            artifact_mask=res["artifact_mask"],
            normal_mask=res["normal_mask"],
            title=f"{tag} (fs={fs} Hz)",
        )
        out_path = out_dir / f"demo_{tag}_artifacts.png"
        fig.savefig(out_path, dpi=160)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

