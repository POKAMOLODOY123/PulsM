"""
Example usage scenarios for the PulseM toolbox.
"""
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from pulsem.data_loader import ECGDataLoader
from pulsem.inference_runner import ECGInference
from pulsem.mobile_converter import MobileModelConverter


def example_data_loading():
    print("=" * 50)
    print("Example 1: Loading and preprocessing")
    print("=" * 50)

    loader = ECGDataLoader(sampling_rate=200)
    signal, sampling_rate = loader.load_csv("2025-10-31_08_47_33.csv")
    print(f"Samples: {len(signal)}, sampling rate: {sampling_rate} Hz, duration: {len(signal) / sampling_rate:.2f}s")

    processed = loader.preprocess_signal(signal)
    print(f"Processed signal -> mean {np.mean(processed):.4f}, std {np.std(processed):.4f}")

    segments = loader.segment_signal(processed, window_size=2000, overlap=0.5)
    print(f"Created {len(segments)} segments of length {len(segments[0])} samples")


def example_inference():
    print("\n" + "=" * 50)
    print("Example 2: Inference pipeline")
    print("=" * 50)

    model_path = Path("artifacts/ecg_artifact_detector.pth")
    if not model_path.exists():
        print("Model not found. Train it first via scripts/train.py")
        return

    loader = ECGDataLoader()
    signal, _ = loader.load_csv("2025-10-31_08_47_33.csv")
    processed = loader.preprocess_signal(signal)
    segments = loader.segment_signal(processed, window_size=2000, overlap=0.5)

    inference = ECGInference(model_path)
    first_result = inference.predict(segments[0])
    print(f"First segment -> {first_result['class_name']} (confidence {first_result['confidence']:.3f})")

    print("\nBatch inference on 5 segments:")
    for i, result in enumerate(inference.predict_batch(segments[:5]), start=1):
        print(f"  Segment {i}: {result['class_name']} ({result['confidence']:.3f})")

    full_results = inference.analyze_signal(signal, window_size=2000, overlap=0.5)
    summary = ECGInference.summarize(full_results)
    print("\nSummary:")
    for class_name, stats in summary["class_distribution"].items():
        print(f"  {class_name}: {stats['count']} ({stats['percentage']:.1f}%)")


def example_synthetic_data():
    print("\n" + "=" * 50)
    print("Example 3: Synthetic data playground")
    print("=" * 50)

    sampling_rate = 200
    duration = 10
    t = np.linspace(0, duration, int(sampling_rate * duration))
    ecg_signal = (
        1.0 * np.sin(2 * np.pi * 1.2 * t)
        + 0.3 * np.sin(2 * np.pi * 2.4 * t)
        + 0.1 * np.random.randn(len(t))
    )

    loader = ECGDataLoader(sampling_rate=sampling_rate)
    processed = loader.preprocess_signal(ecg_signal)
    segments = loader.segment_signal(processed, window_size=2000, overlap=0.5)
    print(f"Generated {len(segments)} synthetic segments")


def example_model_info():
    print("\n" + "=" * 50)
    print("Example 4: Model metadata")
    print("=" * 50)

    model_path = Path("artifacts/ecg_artifact_detector.pth")
    if not model_path.exists():
        print("Model not found.")
        return

    converter = MobileModelConverter(model_path)
    info = converter.model_info()
    for key, value in info.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    print("PulseM usage examples\n")

    try:
        example_data_loading()
        example_inference()
        example_synthetic_data()
        example_model_info()
    except Exception as exc:
        print(f"Example failed: {exc}")

    print("\nDone!")
