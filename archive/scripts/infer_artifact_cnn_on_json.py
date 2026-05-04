"""
Run trained ArtifactCNN on bracelet JSON ECG files and plot Novaroll-like marking.

Input:
  data/saveliy.dubovik@gmail.com/*.json
Model:
  artifacts/artifact_cnn_pseudo.pth
Outputs:
  artifacts/cnn_<filename>.png
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pulsem.data_loader import ECGDataLoader
from pulsem.artifact_cnn import ArtifactCNN, make_windows, zscore
from ecg_plotting import plot_ecg_with_artifacts


def windows_to_point_probs(
    win_probs: np.ndarray, starts: np.ndarray, n_samples: int, window_size: int
) -> np.ndarray:
    """
    Overlap-add: spread window probabilities back to per-point probabilities.
    """
    acc = np.zeros((n_samples,), dtype=np.float32)
    cnt = np.zeros((n_samples,), dtype=np.float32)
    for p, s in zip(win_probs, starts):
        s = int(s)
        e = min(n_samples, s + window_size)
        acc[s:e] += float(p)
        cnt[s:e] += 1.0
    cnt = np.clip(cnt, 1.0, None)
    return acc / cnt


def main() -> None:
    data_dir = PROJECT_ROOT / "data" / "saveliy.dubovik@gmail.com"
    ckpt_path = PROJECT_ROOT / "artifacts" / "artifact_cnn_pseudo.pth"
    out_dir = PROJECT_ROOT / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    window_size = int(ckpt.get("window_size", 500))
    step = int(ckpt.get("step", 250))

    model = ArtifactCNN(window_size=window_size)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    loader = ECGDataLoader()

    for p in sorted(data_dir.glob("*.json")):
        sig, fs = loader.load_json(str(p), default_fs=125)
        if sig.size == 0 or fs != 125:
            continue

        sig_n = zscore(sig)
        windows, starts = make_windows(sig_n, window_size=window_size, step=step)
        if windows.shape[0] == 0:
            continue

        xb = torch.from_numpy(windows[:, None, :])  # (B,1,L)
        with torch.no_grad():
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy().astype(np.float32)

        point_probs = windows_to_point_probs(probs, starts, sig_n.shape[0], window_size)
        artifact_mask = point_probs >= 0.5
        normal_mask = ~artifact_mask

        fig = plot_ecg_with_artifacts(
            sig,
            fs=float(fs),
            artifact_mask=artifact_mask,
            normal_mask=normal_mask,
            title=f"CNN pseudo {p.name}",
        )
        out_path = out_dir / f"cnn_{p.stem}.png"
        fig.savefig(out_path, dpi=140)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

