"""
Plot LabelStudio-trained RF artifact detection on:
  - ЭКГ в покое.json
  - ЭКГ шум.json
  - data/2026-03-31_17_07_37.json
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pulsem.data_loader import ECGDataLoader
from ecg_artifact_filter import ECGArtifactFilter, ECGArtifactFilterConfig
from ecg_plotting import plot_ecg_with_artifacts


def plot_file(filt: ECGArtifactFilter, json_path: Path, out_path: Path, title: str) -> None:
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
    art = res["artifact_mask"]
    print(f"  artifact_fraction={float(art.mean()):.3f}, segments={len(res['artifact_segments'])}")


def main() -> None:
    model_path = PROJECT_ROOT / "artifacts" / "ecg_rf_artifact_model_labelstudio.joblib"
    if not model_path.exists():
        raise RuntimeError(f"Model not found: {model_path}")

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

    out_dir = PROJECT_ROOT / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_file(
        filt,
        PROJECT_ROOT / "ЭКГ в покое.json",
        out_dir / "labelstudio_rf_rest.png",
        "LabelStudio RF - ЭКГ в покое",
    )
    plot_file(
        filt,
        PROJECT_ROOT / "ЭКГ шум.json",
        out_dir / "labelstudio_rf_noisy.png",
        "LabelStudio RF - ЭКГ шум",
    )
    plot_file(
        filt,
        PROJECT_ROOT / "data" / "2026-03-31_17_07_37.json",
        out_dir / "labelstudio_rf_2026-03-31_17_07_37.png",
        "LabelStudio RF - 2026-03-31_17_07_37",
    )


if __name__ == "__main__":
    main()

