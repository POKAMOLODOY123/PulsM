"""
Inference helpers for the ECG artifact detector (PyTorch).
"""
from __future__ import annotations

import argparse
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List

from pulsem.data_loader import ECGDataLoader
from pulsem.model import ECGArtifactDetector, create_lightweight_model
from pulsem.trainer import CLASS_NAMES


class ECGInference:
    """Load a trained model and run predictions on ECG segments."""

    def __init__(self, model_path: Path, device: str | None = None):
        self.model_path = Path(model_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = None
        self.input_length = 2000
        self.num_classes = len(CLASS_NAMES)
        self._load_model()

    def _load_model(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.input_length = checkpoint.get("input_length", 2000)
        self.num_classes = checkpoint.get("num_classes", len(CLASS_NAMES))
        model_cls = create_lightweight_model if checkpoint.get("use_lightweight") else ECGArtifactDetector

        self.model = model_cls(self.input_length, self.num_classes)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"Loaded model from {self.model_path}")
        print(f"Device: {self.device}")
        print(f"Input length: {self.input_length}")

    def predict(self, signal_segment: np.ndarray) -> Dict:
        if isinstance(signal_segment, np.ndarray):
            signal_tensor = torch.FloatTensor(signal_segment)
        else:
            signal_tensor = signal_segment

        if signal_tensor.ndim == 1:
            signal_tensor = signal_tensor.unsqueeze(0)

        signal_tensor = signal_tensor.to(self.device)

        with torch.no_grad():
            outputs = self.model(signal_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()

        return {
            "class": predicted_class,
            "class_name": CLASS_NAMES[predicted_class],
            "confidence": confidence,
            "probabilities": {
                CLASS_NAMES[i]: probabilities[0][i].item() for i in range(probabilities.shape[1])
            },
        }

    def analyze_signal(self, signal_data: np.ndarray, window_size: int = 2000, overlap: float = 0.5) -> List[Dict]:
        loader = ECGDataLoader()
        processed_signal = loader.preprocess_signal(signal_data)
        segments = loader.segment_signal(processed_signal, window_size, overlap)
        return [self.predict(segment) for segment in segments]

    def predict_batch(self, signal_segments: List[np.ndarray]) -> List[Dict]:
        segments_tensor = torch.FloatTensor(np.array(signal_segments)).to(self.device)
        with torch.no_grad():
            outputs = self.model(segments_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_classes = torch.argmax(probabilities, dim=1)

        results = []
        for i in range(len(signal_segments)):
            pred_class = predicted_classes[i].item()
            conf = probabilities[i][pred_class].item()
            results.append(
                {
                    "class": pred_class,
                    "class_name": CLASS_NAMES[pred_class],
                    "confidence": conf,
                    "probabilities": {
                        CLASS_NAMES[j]: probabilities[i][j].item() for j in range(probabilities.shape[1])
                    },
                }
            )
        return results

    @staticmethod
    def summarize(results: List[Dict]) -> Dict:
        total = len(results)
        class_counts = {name: 0 for name in CLASS_NAMES}
        confidences = []

        for result in results:
            class_counts[result["class_name"]] += 1
            confidences.append(result["confidence"])

        return {
            "total_segments": total,
            "class_distribution": {
                name: {"count": count, "percentage": (count / total * 100) if total else 0.0}
                for name, count in class_counts.items()
            },
            "average_confidence": float(np.mean(confidences)) if confidences else 0.0,
            "min_confidence": float(np.min(confidences)) if confidences else 0.0,
            "max_confidence": float(np.max(confidences)) if confidences else 0.0,
        }


def main():
    parser = argparse.ArgumentParser(description="Run inference on ECG signals.")
    parser.add_argument("--model", type=str, default="artifacts/ecg_artifact_detector.pth", help="Path to .pth model.")
    parser.add_argument("--data", type=str, default="2025-10-31_08_47_33.csv", help="Path to raw CSV with ECG values.")
    parser.add_argument("--device", type=str, default=None, help="Inference device (cpu/cuda).")
    parser.add_argument("--window-size", type=int, default=2000)
    parser.add_argument("--overlap", type=float, default=0.5)
    args = parser.parse_args()

    loader = ECGDataLoader()
    signal_data, _ = loader.load_csv(args.data)
    print(f"Loaded {len(signal_data)} samples from {args.data}")

    inference = ECGInference(args.model, device=args.device)
    results = inference.analyze_signal(signal_data, window_size=args.window_size, overlap=args.overlap)

    print(f"\nAnalyzed {len(results)} segments\n")
    for i, result in enumerate(results[:5]):
        print(f"Segment {i + 1}: {result['class_name']} (confidence {result['confidence']:.3f})")
        for class_name, prob in result["probabilities"].items():
            print(f"  {class_name}: {prob:.3f}")

    summary = ECGInference.summarize(results)
    print("\nSummary:")
    print(f"  Total segments: {summary['total_segments']}")
    print(f"  Average confidence: {summary['average_confidence']:.3f}")
    for class_name, stats in summary["class_distribution"].items():
        print(f"  {class_name}: {stats['count']} ({stats['percentage']:.1f}%)")


if __name__ == "__main__":
    main()
