from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class ArtifactCNNConfig:
    fs: int = 125
    window_size: int = 500  # 4 seconds at 125 Hz


class ArtifactCNN(nn.Module):
    """
    Small 1D CNN for window-level artifact classification.

    Input: (batch, 1, window_size)
    Output: logits (batch, 2) for [normal, artifact]
    """

    def __init__(self, window_size: int = 500) -> None:
        super().__init__()
        self.window_size = int(window_size)

        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, padding=4),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return self.head(z)


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    mu = float(np.mean(x)) if x.size else 0.0
    sigma = float(np.std(x)) if x.size else 1.0
    if sigma < 1e-6:
        return x - mu
    return (x - mu) / sigma


def make_windows(x: np.ndarray, window_size: int, step: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Slice 1D signal into overlapping windows.

    Returns:
      windows: (n_windows, window_size)
      starts: (n_windows,) start indices
    """
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    n = x.shape[0]
    if n < window_size:
        return np.empty((0, window_size), dtype=np.float32), np.empty((0,), dtype=np.int64)
    starts = np.arange(0, n - window_size + 1, step, dtype=np.int64)
    windows = np.stack([x[s : s + window_size] for s in starts], axis=0)
    return windows, starts

