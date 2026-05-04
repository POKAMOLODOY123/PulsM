"""
Train a small 1D CNN on bracelet JSON ECG files with pseudo-labels.

Pseudo-labeling:
  - Teacher = OR(heuristic_mask, RF_json_mask)
  - Window label:
      artifact if teacher_artifact_fraction >= 0.50
      normal   if teacher_artifact_fraction <= 0.05
      ignore   otherwise

Input files:
  data/saveliy.dubovik@gmail.com/*.json

Outputs:
  artifacts/artifact_cnn_pseudo.pth
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pulsem.data_loader import ECGDataLoader
from pulsem.artifact_cnn import ArtifactCNN, make_windows, zscore
from ecg_artifact_filter import ECGArtifactFilter, ECGArtifactFilterConfig


class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        x = self.X[idx][None, :]  # (1, L)
        return torch.from_numpy(x), torch.tensor(self.y[idx], dtype=torch.long)


def build_pseudo_dataset(
    folder: Path,
    *,
    window_size: int = 500,
    step: int = 250,
    art_hi: float = 0.50,
    art_lo: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    loader = ECGDataLoader()

    # teacher 1: heuristic
    heur = ECGArtifactFilter(config=ECGArtifactFilterConfig(fs=125))

    # teacher 2: RF trained on your two labeled JSON files (if exists)
    rf_path = PROJECT_ROOT / "artifacts" / "ecg_rf_artifact_model_json.joblib"
    rf = ECGArtifactFilter.from_joblib(str(rf_path), config=ECGArtifactFilterConfig(fs=125))

    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []

    for p in sorted(folder.glob("*.json")):
        sig, fs = loader.load_json(str(p), default_fs=125)
        if sig.size == 0 or fs <= 0:
            continue
        if fs != 125:
            # keep it simple for now: only 125 Hz
            continue

        sig = zscore(sig)

        res_h = heur.infer(sig)
        res_r = rf.infer(sig)
        teacher_mask = np.asarray(res_h["artifact_mask"], dtype=bool) | np.asarray(
            res_r["artifact_mask"], dtype=bool
        )

        windows, starts = make_windows(sig, window_size=window_size, step=step)
        if windows.shape[0] == 0:
            continue

        labels = np.full((windows.shape[0],), fill_value=-1, dtype=np.int64)
        for i, s in enumerate(starts):
            w_mask = teacher_mask[int(s) : int(s) + window_size]
            frac = float(np.mean(w_mask)) if w_mask.size else 0.0
            if frac >= art_hi:
                labels[i] = 1
            elif frac <= art_lo:
                labels[i] = 0

        keep = labels >= 0
        if not np.any(keep):
            continue

        X_list.append(windows[keep])
        y_list.append(labels[keep])

        frac_kept = float(np.mean(labels[keep] == 1))
        print(f"{p.name}: kept_windows={int(keep.sum())}, artifact_frac_in_kept={frac_kept:.3f}")

    if not X_list:
        raise RuntimeError("No training windows created (check inputs / thresholds)")

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    print(f"Total windows: {X.shape[0]}, artifact_fraction={float(np.mean(y==1)):.3f}")
    return X, y


def main() -> None:
    data_dir = PROJECT_ROOT / "data" / "saveliy.dubovik@gmail.com"
    out_path = PROJECT_ROOT / "artifacts" / "artifact_cnn_pseudo.pth"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    window_size = 500
    step = 250

    X, y = build_pseudo_dataset(data_dir, window_size=window_size, step=step)

    # split
    rng = np.random.default_rng(42)
    idx = np.arange(X.shape[0])
    rng.shuffle(idx)
    split = int(0.8 * len(idx))
    tr_idx, va_idx = idx[:split], idx[split:]
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_va, y_va = X[va_idx], y[va_idx]

    train_loader = DataLoader(WindowDataset(X_tr, y_tr), batch_size=256, shuffle=True)
    val_loader = DataLoader(WindowDataset(X_va, y_va), batch_size=256, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ArtifactCNN(window_size=window_size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

    # class imbalance
    pos = float(np.sum(y_tr == 1))
    neg = float(np.sum(y_tr == 0))
    w_pos = neg / max(pos, 1.0)
    class_weight = torch.tensor([1.0, w_pos], dtype=torch.float32, device=device)

    best_val = -1.0
    for epoch in range(1, 6):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb, weight=class_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * xb.shape[0]

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                pred = torch.argmax(logits, dim=1)
                correct += int((pred == yb).sum().item())
                total += int(yb.shape[0])
        acc = correct / max(total, 1)
        avg_loss = total_loss / max(len(train_loader.dataset), 1)
        print(f"epoch={epoch} train_loss={avg_loss:.4f} val_acc={acc:.4f}")

        if acc > best_val:
            best_val = acc
            torch.save(
                {"state_dict": model.state_dict(), "window_size": window_size, "step": step},
                out_path,
            )

    print(f"Saved model to {out_path} (best_val_acc={best_val:.4f})")


if __name__ == "__main__":
    main()

