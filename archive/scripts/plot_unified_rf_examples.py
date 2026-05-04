"""
Plot unified 125 Hz RF artifact detection on key example files:
  - ЭКГ в покое.json
  - ЭКГ шум.json
  - one 125 Hz file from data/saveliy.dubovik@gmail.com
  - one 125 Hz file from data/German

Outputs:
  artifacts/unified_rf_rest.png
  artifacts/unified_rf_noisy.png
  artifacts/unified_rf_saveliy_sample.png
  artifacts/unified_rf_german_sample.png
"""

from __future__ import annotations

from pathlib import Path
import sys

from typing import Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pulsem.data_loader import ECGDataLoader
from ecg_artifact_filter import ECGArtifactFilter, ECGArtifactFilterConfig
from ecg_plotting import plot_ecg_with_artifacts


def find_first_125hz_json(folder: Path) -> Optional[Path]:
    loader = ECGDataLoader()
    for p in sorted(folder.glob("*.json")):
        signal, fs = loader.load_json(str(p), default_fs=125)
        if signal.size > 0 and fs == 125:
            return p
    return None


def plot_file(
    filt: ECGArtifactFilter,
    json_path: Path,
    out_path: Path,
    title: str,
) -> None:
    loader = ECGDataLoader()
    signal, fs = loader.load_json(str(json_path), default_fs=125)
    res = filt.infer(signal)
    fig = plot_ecg_with_artifacts(
        signal,
        fs=float(fs),
        artifact_mask=res["artifact_mask"],
        normal_mask=res["normal_mask"],
        title=title,
    )
    fig.savefig(out_path, dpi=160)
    print(f"Saved: {out_path}")


def main() -> None:
    model_path = PROJECT_ROOT / "artifacts" / "ecg_rf_artifact_model_125_unified.joblib"
    filt = ECGArtifactFilter.from_joblib(
        str(model_path),
        config=ECGArtifactFilterConfig(
            fs=125,
            prob_threshold=0.7,
            prob_smooth_window=101,
            min_artifact_duration=125,
            gap_tolerance=30,
            use_baseline_deviation_rule=True,
            baseline_window=125,
            baseline_std_k=3.0,
            # Plateau/flatline rules are too aggressive on quantized clean ECG;
            # keep them off in default production-like plotting config.
            use_plateau_rule=False,
            use_flatline_rule=False,
            normalize_input=True,
            include_variance=True,
            use_multi_scale=True,
            multi_scale_min_durations=(50, 125, 188, 250),
        ),
    )

    out_dir = PROJECT_ROOT / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) ЭКГ в покое
    rest_path = PROJECT_ROOT / "ЭКГ в покое.json"
    plot_file(
        filt,
        rest_path,
        out_dir / "unified_rf_rest.png",
        "Unified RF – ЭКГ в покое",
    )

    # 2) ЭКГ шум
    noisy_path = PROJECT_ROOT / "ЭКГ шум.json"
    plot_file(
        filt,
        noisy_path,
        out_dir / "unified_rf_noisy.png",
        "Unified RF – ЭКГ шум",
    )

    # 3) Один 125 Hz файл из saveliy папки
    sav_dir = PROJECT_ROOT / "data" / "saveliy.dubovik@gmail.com"
    sav_sample = find_first_125hz_json(sav_dir)
    if sav_sample is not None:
        plot_file(
            filt,
            sav_sample,
            out_dir / "unified_rf_saveliy_sample.png",
            f"Unified RF – saveliy sample ({sav_sample.name})",
        )
    else:
        print("No 125Hz saveliy JSON file found")

    # 4) Один 125 Hz файл из German
    ger_dir = PROJECT_ROOT / "data" / "German"
    ger_sample = find_first_125hz_json(ger_dir)
    if ger_sample is not None:
        plot_file(
            filt,
            ger_sample,
            out_dir / "unified_rf_german_sample.png",
            f"Unified RF – German sample ({ger_sample.name})",
        )
    else:
        print("No 125Hz German JSON file found")

    # 5) Новый файл из ./data
    new_path = PROJECT_ROOT / "data" / "2026-03-31_17_07_37.json"
    if new_path.exists():
        plot_file(
            filt,
            new_path,
            out_dir / "unified_rf_new_2026-03-31_17_07_37.png",
            "Unified RF – new file (2026-03-31_17_07_37)",
        )
    else:
        print("New file not found in ./data")


if __name__ == "__main__":
    main()

