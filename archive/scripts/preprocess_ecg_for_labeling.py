"""
Python alternative to preprocess.m for ECG JSON files.

What it does:
1) Loads ECG from JSON: {"values": [...], "samplingRateHz": ...}
2) Filters signal for display (0.5-20 Hz)
3) Detects R-peaks (simple robust pipeline)
4) Estimates HR from RR intervals
5) Builds initial quality mask (good/bad) to speed up labeling
6) Saves plots + CSV for Label Studio

Usage:
  python scripts/preprocess_ecg_for_labeling.py --input "data/2026-03-31_17_07_37.json"
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks


@dataclass
class PreprocessConfig:
    lowcut_hz: float = 0.5
    highcut_hz: float = 20.0
    filter_order: int = 4
    min_rr_s: float = 0.28   # ~214 bpm upper physiological bound
    max_rr_s: float = 1.8    # ~33 bpm lower physiological bound
    hr_min: float = 40.0
    hr_max: float = 190.0
    flatline_std_thr: float = 0.015
    flatline_win_s: float = 1.0
    bad_expansion_s: float = 0.4
    peak_match_tol_s: float = 0.12
    lost_peak_expansion_s: float = 0.35


def load_ecg_json(path: Path) -> tuple[np.ndarray, int]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    values = np.asarray(data.get("values", []), dtype=np.float64)
    fs = int(data.get("samplingRateHz", 125))
    return values, fs


def bandpass_filter(x: np.ndarray, fs: int, low: float, high: float, order: int) -> np.ndarray:
    if x.size == 0:
        return x.astype(np.float64)
    nyq = fs / 2.0
    low_n = max(1e-5, low / nyq)
    high_n = min(0.999, high / nyq)
    if low_n >= high_n:
        return x.astype(np.float64)
    b, a = butter(order, [low_n, high_n], btype="bandpass")
    return filtfilt(b, a, x).astype(np.float64)


def robust_normalize(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x.astype(np.float64)
    x0 = x - np.median(x)
    s = float(np.std(x0))
    if s < 1e-8:
        return x0
    return x0 / s


def detect_r_peaks(filtered_norm: np.ndarray, fs: int) -> np.ndarray:
    """
    Lightweight peak detector:
    - square the signal (energy)
    - smooth over ~120 ms
    - adaptive threshold from percentile
    - enforce min distance between peaks
    """
    if filtered_norm.size == 0:
        return np.array([], dtype=np.int64)

    energy = filtered_norm * filtered_norm
    win = max(3, int(0.12 * fs))
    kernel = np.ones(win, dtype=np.float64) / float(win)
    smooth = np.convolve(energy, kernel, mode="same")

    thr = np.percentile(smooth, 92)
    min_distance = max(1, int(0.28 * fs))
    peaks, _ = find_peaks(smooth, height=thr, distance=min_distance)
    return peaks.astype(np.int64)


def find_lost_peaks(raw_peaks: np.ndarray, filt_peaks: np.ndarray, fs: int, tol_s: float) -> np.ndarray:
    """
    Peaks present in raw-domain detector but absent in filtered-domain detector.
    """
    if raw_peaks.size == 0:
        return np.array([], dtype=np.int64)
    if filt_peaks.size == 0:
        return raw_peaks.astype(np.int64)

    tol = int(max(1, round(tol_s * fs)))
    lost = []
    j = 0
    f = filt_peaks.astype(np.int64)
    for rp in raw_peaks.astype(np.int64):
        while j < f.size and f[j] < rp - tol:
            j += 1
        matched = False
        if j < f.size and abs(int(f[j]) - int(rp)) <= tol:
            matched = True
        elif j > 0 and abs(int(f[j - 1]) - int(rp)) <= tol:
            matched = True
        if not matched:
            lost.append(int(rp))
    return np.asarray(lost, dtype=np.int64)


def rolling_std(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or x.size == 0:
        return np.zeros_like(x)
    kernel = np.ones(w, dtype=np.float64) / float(w)
    m1 = np.convolve(x, kernel, mode="same")
    m2 = np.convolve(x * x, kernel, mode="same")
    var = np.clip(m2 - m1 * m1, a_min=0.0, a_max=None)
    return np.sqrt(var)


def build_quality_mask(
    x_norm: np.ndarray,
    peaks: np.ndarray,
    lost_peaks: np.ndarray,
    fs: int,
    cfg: PreprocessConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      good_mask (bool), t_hr (float array), hr (float array)
    """
    n = x_norm.size
    bad = np.zeros(n, dtype=bool)

    # 1) Flatline-like areas
    std_w = rolling_std(x_norm, max(3, int(cfg.flatline_win_s * fs)))
    bad |= std_w < cfg.flatline_std_thr

    # 2) HR/RR plausibility around detected beats
    if peaks.size >= 2:
        t_peaks = peaks / float(fs)
        rr = np.diff(t_peaks)
        hr = 60.0 / np.maximum(rr, 1e-6)
        t_hr = (t_peaks[:-1] + t_peaks[1:]) / 2.0

        bad_rr = (rr < cfg.min_rr_s) | (rr > cfg.max_rr_s)
        bad_hr = (hr < cfg.hr_min) | (hr > cfg.hr_max)
        bad_idx = np.where(bad_rr | bad_hr)[0]

        expand = int(cfg.bad_expansion_s * fs)
        for i in bad_idx:
            s = max(0, peaks[i] - expand)
            e = min(n, peaks[i + 1] + expand)
            bad[s:e] = True
    else:
        t_hr = np.array([], dtype=np.float64)
        hr = np.array([], dtype=np.float64)
        bad[:] = True

    # 3) If R-peak was present on raw but disappeared on filtered, mark around it as bad.
    if lost_peaks.size > 0:
        exp_lost = int(cfg.lost_peak_expansion_s * fs)
        for p in lost_peaks:
            s = max(0, int(p) - exp_lost)
            e = min(n, int(p) + exp_lost)
            bad[s:e] = True

    good = ~bad
    return good, t_hr, hr


def save_plots(
    out_dir: Path,
    stem: str,
    t: np.ndarray,
    raw: np.ndarray,
    disp: np.ndarray,
    peaks_filtered: np.ndarray,
    peaks_raw: np.ndarray,
    lost_peaks: np.ndarray,
    t_hr: np.ndarray,
    hr: np.ndarray,
    good_mask: np.ndarray,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Raw + peaks, filtered + good/bad overlay
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    ax1, ax2 = axes
    ax1.plot(t, raw, color="black", linewidth=0.7, label="raw")
    if peaks_raw.size > 0:
        ax1.plot(t[peaks_raw], raw[peaks_raw], "go", markersize=3, label="R-peaks raw")
    if lost_peaks.size > 0:
        ax1.plot(t[lost_peaks], raw[lost_peaks], "rx", markersize=5, label="Lost after filtering")
    ax1.set_title("Raw ECG + peaks (green) + lost peaks (red x)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")

    ax2.plot(t, disp, color="#1f77b4", linewidth=0.8, label="filtered 0.5-20 Hz")
    if peaks_filtered.size > 0:
        ax2.plot(t[peaks_filtered], disp[peaks_filtered], "ro", markersize=3, label="R-peaks filtered")
    # overlay bad mask in red
    bad = ~good_mask
    if bad.any():
        diff = np.diff(bad.astype(int))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0]
        if bad[0]:
            starts = np.concatenate(([0], starts))
        if bad[-1]:
            ends = np.concatenate((ends, [bad.size - 1]))
        for s, e in zip(starts, ends):
            ax2.axvspan(t[s], t[e], color="#ff3333", alpha=0.2, linewidth=0)
    ax2.set_title("Filtered ECG + bad-quality regions + filtered peaks")
    ax2.set_xlabel("Time, s")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}_ecg_peaks_quality.png", dpi=150)
    plt.close(fig)

    # Plot 2: HR
    fig = plt.figure(figsize=(12, 4))
    if t_hr.size > 0:
        plt.plot(t_hr, hr, "-o", markersize=3)
    plt.title("Heart Rate from RR intervals")
    plt.xlabel("Time, s")
    plt.ylabel("HR, bpm")
    plt.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}_hr.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to ECG JSON file")
    parser.add_argument("--output-dir", default="artifacts/label_prep", help="Output directory")
    args = parser.parse_args()

    cfg = PreprocessConfig()
    inp = Path(args.input)
    out_dir = Path(args.output_dir)
    stem = inp.stem

    raw, fs = load_ecg_json(inp)
    if raw.size == 0:
        raise RuntimeError(f"Empty ECG in {inp}")

    t = np.arange(raw.size, dtype=np.float64) / float(fs)
    disp = bandpass_filter(raw, fs, cfg.lowcut_hz, cfg.highcut_hz, cfg.filter_order)
    disp_norm = robust_normalize(disp)
    raw_norm = robust_normalize(raw)
    peaks_filtered = detect_r_peaks(disp_norm, fs)
    peaks_raw = detect_r_peaks(raw_norm, fs)
    lost_peaks = find_lost_peaks(peaks_raw, peaks_filtered, fs, cfg.peak_match_tol_s)
    good_mask, t_hr, hr = build_quality_mask(disp_norm, peaks_filtered, lost_peaks, fs, cfg)

    # CSV for Label Studio / manual review
    df = pd.DataFrame(
        {
            "time": t,
            "ecg_raw": raw,
            "ecg_filtered": disp,
            "r_peak_filtered": 0,
            "r_peak_raw": 0,
            "r_peak_lost_after_filter": 0,
            "quality_good": good_mask.astype(int),
            "quality_bad": (~good_mask).astype(int),
        }
    )
    if peaks_filtered.size > 0:
        df.loc[peaks_filtered, "r_peak_filtered"] = 1
    if peaks_raw.size > 0:
        df.loc[peaks_raw, "r_peak_raw"] = 1
    if lost_peaks.size > 0:
        df.loc[lost_peaks, "r_peak_lost_after_filter"] = 1
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{stem}_label_prep.csv"
    df.to_csv(csv_path, index=False)

    save_plots(
        out_dir,
        stem,
        t,
        raw,
        disp,
        peaks_filtered,
        peaks_raw,
        lost_peaks,
        t_hr,
        hr,
        good_mask,
    )

    print(f"Saved CSV: {csv_path}")
    print(f"Saved plots to: {out_dir}")
    print(
        f"fs={fs}, n={raw.size}, peaks_raw={peaks_raw.size}, "
        f"peaks_filtered={peaks_filtered.size}, lost_peaks={lost_peaks.size}, "
        f"bad_fraction={(~good_mask).mean():.3f}"
    )


if __name__ == "__main__":
    main()

