"""
Training pipeline for the ECG artifact detector (PyTorch).
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset, random_split

from pulsem.data_loader import ECGDataLoader
from pulsem.model import ECGArtifactDetector, create_lightweight_model

CLASS_NAMES = ["Normal", "Mechanical Artifact", "Heart Issue"]


class ECGDataset(Dataset):
    """Torch dataset that stores ECG segments and their labels."""

    def __init__(self, signals: np.ndarray, labels: np.ndarray):
        self.signals = torch.FloatTensor(signals)
        self.labels = torch.LongTensor(labels)

    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(self, idx: int):
        return self.signals[idx], self.labels[idx]


class EarlyStopping:
    """Utility for interrupting training when the validation loss stagnates."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss: Optional[float] = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_weights = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False


class ECGTrainer:
    """Wraps model creation, training loop, and evaluation helpers."""

    def __init__(
        self,
        input_length: int = 2000,
        num_classes: int = 3,
        use_lightweight: bool = False,
        device: Optional[str] = None,
    ):
        self.input_length = input_length
        self.num_classes = num_classes
        self.use_lightweight = use_lightweight
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        print(f"Using device: {self.device}")

        model_cls = create_lightweight_model if use_lightweight else ECGArtifactDetector
        self.model = model_cls(input_length, num_classes).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = None
        self.history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    def prepare_data(
        self,
        signal_data: np.ndarray,
        labels: np.ndarray,
        test_size: float = 0.2,
        val_size: float = 0.1,
        batch_size: int = 32,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        dataset = ECGDataset(signal_data, labels)

        n_samples = len(dataset)
        n_test = int(n_samples * test_size)
        n_val = int(n_samples * val_size)
        n_train = n_samples - n_test - n_val

        train_dataset, val_dataset, test_dataset = random_split(
            dataset,
            [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(42),
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, test_loader

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        for signals, labels in train_loader:
            signals = signals.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(signals)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        return avg_loss, accuracy

    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for signals, labels in val_loader:
                signals = signals.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(signals)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        return avg_loss, accuracy

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        early_stopping_patience: int = 10,
        use_scheduler: bool = True,
        best_model_path: Optional[Path] = None,
    ) -> dict:
        if use_scheduler:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
            )

        early_stopping = EarlyStopping(patience=early_stopping_patience)
        best_val_loss = float("inf")

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)

            if use_scheduler:
                self.scheduler.step(val_loss)

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            print(f"Epoch [{epoch + 1}/{epochs}]")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if best_model_path is not None:
                    best_model_path.parent.mkdir(parents=True, exist_ok=True)
                    self.save_model(best_model_path)

            if early_stopping(val_loss, self.model):
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

        return self.history

    def evaluate(self, test_loader: DataLoader) -> dict:
        self.model.eval()
        all_preds, all_labels = [], []
        total_loss = 0.0

        with torch.no_grad():
            for signals, labels in test_loader:
                signals = signals.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(signals)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        avg_loss = total_loss / len(test_loader)
        # Get unique classes present in the data
        unique_classes = sorted(list(set(all_labels + all_preds)))
        class_names_subset = [CLASS_NAMES[i] for i in unique_classes if i < len(CLASS_NAMES)]
        
        report = classification_report(
            all_labels, all_preds, 
            labels=unique_classes,
            target_names=class_names_subset
        )
        cm = confusion_matrix(all_labels, all_preds).tolist()

        return {"loss": avg_loss, "accuracy": accuracy, "classification_report": report, "confusion_matrix": cm}

    def save_model(self, filepath: Path) -> None:
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "input_length": self.input_length,
            "num_classes": self.num_classes,
            "use_lightweight": self.use_lightweight,
            "history": self.history,
        }
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: Path) -> None:
        checkpoint = torch.load(filepath, map_location=self.device)
        model_cls = create_lightweight_model if checkpoint.get("use_lightweight") else ECGArtifactDetector
        self.model = model_cls(checkpoint["input_length"], checkpoint["num_classes"])
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.history = checkpoint.get("history", {})
        print(f"Model loaded from {filepath}")


def load_npz_dataset(dataset_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(dataset_path)
    return data["signals"], data["labels"]


def main():
    parser = argparse.ArgumentParser(description="Train the ECG artifact detector.")
    parser.add_argument("--dataset", type=str, default="data/processed/mit_bih_segments.npz", help="Path to .npz dataset.")
    parser.add_argument(
        "--csv",
        type=str,
        default="2025-10-31_08_47_33.csv",
        help="Fallback CSV file with raw ECG signal (used when dataset is missing).",
    )
    parser.add_argument("--sampling-rate", type=int, default=200, help="Sampling rate for raw CSV ingestion.")
    parser.add_argument("--input-length", type=int, default=2000, help="Number of samples per segment.")
    parser.add_argument("--overlap", type=float, default=0.5, help="Overlap ratio for raw segmentation.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lightweight", action="store_true", help="Use the mobile-friendly model.")
    parser.add_argument("--device", type=str, default=None, help="Force device (cpu/cuda).")
    parser.add_argument("--output-dir", type=str, default="artifacts", help="Directory for models and metrics.")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    signals: np.ndarray
    labels: np.ndarray

    if dataset_path.exists():
        print(f"Loading labeled dataset from {dataset_path}")
        signals, labels = load_npz_dataset(dataset_path)
    else:
        print(f"No dataset found at {dataset_path}, falling back to raw CSV {args.csv}")
        loader = ECGDataLoader(sampling_rate=args.sampling_rate)
        raw_signal, sampling_rate = loader.load_csv(args.csv)
        print(f"Loaded {len(raw_signal)} samples at {sampling_rate} Hz")
        processed_signal = loader.preprocess_signal(raw_signal)
        segments = loader.segment_signal(processed_signal, window_size=args.input_length, overlap=args.overlap)
        signals = np.stack(segments)
        labels = np.random.randint(0, 3, size=len(signals))  # placeholder until real labels are provided

    trainer = ECGTrainer(
        input_length=args.input_length,
        num_classes=len(CLASS_NAMES),
        use_lightweight=args.lightweight,
        device=args.device,
    )

    train_loader, val_loader, test_loader = trainer.prepare_data(
        signals, labels, batch_size=args.batch_size
    )

    print(f"Training/Validation/Test sizes: {len(train_loader.dataset)}/{len(val_loader.dataset)}/{len(test_loader.dataset)}")

    output_dir = Path(args.output_dir)
    model_path = output_dir / "ecg_artifact_detector.pth"
    history = trainer.train(
        train_loader,
        val_loader,
        epochs=args.epochs,
        early_stopping_patience=10,
        best_model_path=model_path,
    )

    metrics = trainer.evaluate(test_loader)
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print(metrics["classification_report"])

    metrics_path = output_dir / "training_metrics.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as fp:
        json.dump(
            {
                "accuracy": float(metrics["accuracy"]),
                "loss": float(metrics["loss"]),
                "confusion_matrix": metrics["confusion_matrix"],
                "history": history,
            },
            fp,
            indent=2,
        )
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
