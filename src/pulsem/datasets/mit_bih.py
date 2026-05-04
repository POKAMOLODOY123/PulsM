"""
Utilities for downloading and preprocessing MIT-BIH ECG datasets.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from scipy.signal import resample
import wfdb
from tqdm import tqdm


ARRHYTHMIA_DB_NAME = "mitdb"
NOISE_DB_NAME = "nstdb"

ARR_RECORDS = [
    "100",
    "101",
    "102",
    "103",
    "104",
    "105",
    "106",
    "107",
    "108",
    "109",
    "111",
    "112",
    "113",
    "114",
    "115",
    "116",
    "117",
    "118",
    "119",
    "121",
    "122",
    "123",
    "124",
    "200",
    "201",
    "202",
    "203",
    "205",
    "207",
    "208",
    "209",
    "210",
    "212",
    "213",
    "214",
    "215",
    "217",
    "219",
    "220",
    "221",
    "222",
    "223",
    "228",
    "230",
    "231",
    "232",
    "233",
    "234",
]

NOISE_RECORDS = [
    "118e24",
    "118e30",
    "119e24",
    "119e30",
    "201e24",
    "201e30",
    "202e24",
    "202e30",
    "207e24",
    "207e30",
    "208e24",
    "208e30",
]

NORMAL_BEAT_SYMBOLS = {"N", "L", "R", "e", "j", "A"}


def download_database(db_name: str, output_dir: Path) -> None:
    """
    Download a PhysioNet database if it is not already present locally.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.glob("*")):
        return

    wfdb.dl_database(db_name, dl_dir=str(output_dir))


def _resample_segment(segment: np.ndarray, target_length: int) -> np.ndarray:
    if len(segment) == target_length:
        return segment.astype(np.float32)
    return resample(segment, target_length).astype(np.float32)


def _segment_signal(
    signal: np.ndarray,
    window_size: int,
    step: int,
    target_length: int,
) -> Iterable[Tuple[np.ndarray, int]]:
    for start in range(0, len(signal) - window_size + 1, step):
        window = signal[start : start + window_size]
        yield _resample_segment(window, target_length), start


def _labels_from_annotations(
    ann_samples: np.ndarray,
    ann_symbols: List[str],
    start_idx: int,
    end_idx: int,
) -> int:
    mask = (ann_samples >= start_idx) & (ann_samples < end_idx)
    if not np.any(mask):
        return 0  # Normal
    symbols = {ann_symbols[i] for i, flag in enumerate(mask) if flag}
    if symbols.difference(NORMAL_BEAT_SYMBOLS):
        return 2  # Heart issue / arrhythmia
    return 0


def extract_arrhythmia_segments(
    db_dir: Path,
    window_duration_s: float = 10.0,
    step_ratio: float = 0.5,
    target_length: int = 2000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Slice the MIT-BIH Arrhythmia database into labeled segments.
    """
    segments: List[np.ndarray] = []
    labels: List[int] = []

    for record in tqdm(ARR_RECORDS, desc="Arrhythmia records"):
        record_path = db_dir / record
        if not record_path.with_suffix(".dat").exists():
            continue

        rec = wfdb.rdrecord(str(record_path))
        ann = wfdb.rdann(str(record_path), "atr")

        signal = rec.p_signal[:, 0]
        fs = int(rec.fs)
        window_size = int(window_duration_s * fs)
        step = max(int(window_size * (1 - step_ratio)), 1)

        for window, start in _segment_signal(signal, window_size, step, target_length):
            end = start + window_size
            label = _labels_from_annotations(np.array(ann.sample), ann.symbol, start, end)
            segments.append(window)
            labels.append(label)

    return np.array(segments), np.array(labels)


def extract_noise_segments(
    db_dir: Path,
    window_duration_s: float = 10.0,
    step_ratio: float = 0.5,
    target_length: int = 2000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Slice the Noise Stress Test database. Entire windows are labeled as mechanical artifacts.
    """
    segments: List[np.ndarray] = []
    labels: List[int] = []

    for record in tqdm(NOISE_RECORDS, desc="Noise stress records"):
        record_path = db_dir / record
        if not record_path.with_suffix(".dat").exists():
            continue

        rec = wfdb.rdrecord(str(record_path))
        signal = rec.p_signal[:, 0]
        fs = int(rec.fs)
        window_size = int(window_duration_s * fs)
        step = max(int(window_size * (1 - step_ratio)), 1)

        for window, _ in _segment_signal(signal, window_size, step, target_length):
            segments.append(window)
            labels.append(1)  # Mechanical artifact

    return np.array(segments), np.array(labels)


def prepare_mit_bih_datasets(
    raw_root: Path,
    processed_root: Path,
    window_duration_s: float = 10.0,
    target_length: int = 2000,
    step_ratio: float = 0.5,
) -> Path:
    """
    Download MIT-BIH datasets and create a combined labeled dataset with synthetic noise.
    """
    from .synthetic_noise import create_balanced_dataset
    
    arr_dir = raw_root / "mit_bih_arrhythmia"
    processed_root.mkdir(parents=True, exist_ok=True)

    download_database(ARRHYTHMIA_DB_NAME, arr_dir)
    print("Skipping Noise Stress Test Database (access issues)")

    arr_segments, arr_labels = extract_arrhythmia_segments(
        arr_dir, window_duration_s, step_ratio, target_length
    )
    
    # Generate synthetic mechanical artifacts
    print("Generating synthetic mechanical artifacts...")
    normal_mask = arr_labels == 0
    heart_issue_mask = arr_labels == 2
    
    normal_segments = arr_segments[normal_mask]
    heart_issue_segments = arr_segments[heart_issue_mask]
    
    signals, labels = create_balanced_dataset(normal_segments, heart_issue_segments)

    dataset_path = processed_root / "mit_bih_segments.npz"
    np.savez_compressed(
        dataset_path, 
        signals=signals, 
        labels=labels,
        class_names=["Normal", "Mechanical Artifact", "Heart Issue"]
    )
    
    print(f"Dataset contains {len(signals)} segments:")
    unique, counts = np.unique(labels, return_counts=True)
    class_names = ["Normal", "Mechanical Artifact", "Heart Issue"]
    for label, count in zip(unique, counts):
        print(f"  {class_names[int(label)]}: {count} segments")

    return dataset_path

