"""
Generate a debug plot for 'ЭКГ шум.json' showing:
 - ECG with artifact mask overlay
 - artifact probabilities (raw + smoothed)
 - selected features (std/var/mean|grad|) for a chosen window.
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
from ecg_artifact_filter import ECGArtifactFilter, ECGArtifactFilterConfig, _rolling_mean, _rolling_std
from ecg_plotting import plot_ecg_debug


def main() -> None:
    loader = ECGDataLoader()
    noisy_path = PROJECT_ROOT / "ЭКГ шум.json"
    signal, fs = loader.load_json(str(noisy_path), default_fs=125)

    model_path = PROJECT_ROOT / "artifacts" / "ecg_rf_artifact_model_125_unified.joblib"
    cfg = ECGArtifactFilterConfig(
        fs=125,
        prob_threshold=0.6,
        prob_smooth_window=101,
        min_artifact_duration=125,
        gap_tolerance=30,
        normalize_input=True,
        include_variance=True,
        use_multi_scale=True,
        multi_scale_min_durations=(50, 125, 188, 250),
    )
    filt = ECGArtifactFilter.from_joblib(str(model_path), config=cfg)
    res = filt.infer(signal)

    probs = np.asarray(res["artifact_probs"], dtype=float)
    probs_sm = _rolling_mean(probs, cfg.prob_smooth_window) if cfg.prob_smooth_window > 1 else probs

    # Build simple feature traces for explanation (on normalized signal)
    x = signal.astype(float)
    x = x - np.median(x)
    s = float(np.std(x))
    if s > 1e-6:
        x = x / s
    grad = np.empty_like(x)
    if x.size > 1:
        grad[0] = x[1] - x[0]
        grad[1:] = x[1:] - x[:-1]
    else:
        grad.fill(0.0)
    abs_grad = np.abs(grad)

    w = 188  # ~1500ms at 125Hz
    feat_std = _rolling_std(x, w)
    feat_var = feat_std * feat_std
    feat_mean_abs_grad = _rolling_mean(abs_grad, w)

    fig = plot_ecg_debug(
        signal,
        fs=float(fs),
        artifact_mask=res["artifact_mask"],
        normal_mask=res["normal_mask"],
        artifact_probs=probs,
        artifact_probs_smoothed=probs_sm,
        feature_std=feat_std,
        feature_var=feat_var,
        feature_mean_abs_grad=feat_mean_abs_grad,
        title="Debug: ЭКГ шум (prob + features)",
    )

    out_path = PROJECT_ROOT / "artifacts" / "debug_noisy_prob_features.png"
    fig.savefig(out_path, dpi=160)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

