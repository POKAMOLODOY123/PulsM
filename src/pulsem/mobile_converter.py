"""
Convert trained PyTorch models into ONNX and TorchScript formats.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from pulsem.model import ECGArtifactDetector, create_lightweight_model


class MobileModelConverter:
    """Utility wrapper that loads a checkpoint and exports it to mobile-friendly formats."""

    def __init__(self, model_path: Path, device: Optional[str] = None):
        self.model_path = Path(model_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model: Optional[torch.nn.Module] = None
        self.input_length = 2000
        self.num_classes = 3
        self._load_model()

    def _load_model(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.input_length = checkpoint.get("input_length", 2000)
        self.num_classes = checkpoint.get("num_classes", 3)
        model_cls = create_lightweight_model if checkpoint.get("use_lightweight") else ECGArtifactDetector
        self.model = model_cls(self.input_length, self.num_classes)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device).eval()

        print(f"Loaded model from {self.model_path}")
        print(f"Input length: {self.input_length}, classes: {self.num_classes}, device: {self.device}")

    def convert_to_onnx(self, output_path: Path, opset_version: int = 11, dynamic_axes: bool = False) -> Path:
        print("Exporting to ONNX...")
        dummy_input = torch.randn(1, self.input_length).to(self.device)
        dynamic = {"input": {0: "batch_size"}, "output": {0: "batch_size"}} if dynamic_axes else None

        torch.onnx.export(
            self.model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic,
        )
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"Saved ONNX model to {output_path} ({size_mb:.2f} MB)")
        return output_path

    def convert_to_torchscript(self, output_path: Path, method: str = "trace") -> Path:
        print(f"Exporting to TorchScript via {method}...")
        if method == "trace":
            dummy_input = torch.randn(1, self.input_length).to(self.device)
            traced_model = torch.jit.trace(self.model, dummy_input)
            traced_model.save(str(output_path))
        elif method == "script":
            scripted_model = torch.jit.script(self.model)
            scripted_model.save(str(output_path))
        else:
            raise ValueError("Method must be 'trace' or 'script'")

        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"Saved TorchScript model to {output_path} ({size_mb:.2f} MB)")
        return output_path

    def quantize_model(self, method: str = "dynamic", representative_data: Optional[np.ndarray] = None) -> torch.nn.Module:
        print(f"Quantizing model with {method} method...")
        if method == "dynamic":
            quantized_model = torch.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear, torch.nn.Conv1d}, dtype=torch.qint8
            )
        elif method == "static":
            if representative_data is None:
                raise ValueError("Static quantization requires representative_data")
            self.model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
            torch.quantization.prepare(self.model, inplace=True)
            with torch.no_grad():
                for data in representative_data[:100]:
                    input_tensor = torch.FloatTensor(data).unsqueeze(0).to(self.device)
                    _ = self.model(input_tensor)
            quantized_model = torch.quantization.convert(self.model, inplace=False)
        else:
            raise ValueError("Unknown quantization method")

        print("Quantization complete.")
        return quantized_model

    def convert_quantized_to_onnx(
        self,
        output_path: Path,
        representative_data: Optional[np.ndarray] = None,
    ) -> Path:
        quantized_model = self.quantize_model("dynamic", representative_data)
        dummy_input = torch.randn(1, self.input_length)
        torch.onnx.export(
            quantized_model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
        )
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"Saved quantized ONNX model to {output_path} ({size_mb:.2f} MB)")
        return output_path

    def model_info(self) -> dict:
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {
            "input_length": self.input_length,
            "num_classes": self.num_classes,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024),
        }


def main():
    parser = argparse.ArgumentParser(description="Convert the trained model to ONNX/TorchScript.")
    parser.add_argument("--model", type=str, default="artifacts/ecg_artifact_detector.pth")
    parser.add_argument("--output-onnx", type=str, default="artifacts/ecg_model.onnx")
    parser.add_argument("--output-torchscript", type=str, default="artifacts/ecg_model.pt")
    parser.add_argument("--quantize", action="store_true", help="Export an additional quantized TorchScript model.")
    parser.add_argument("--method", choices=["trace", "script"], default="trace")
    args = parser.parse_args()

    converter = MobileModelConverter(args.model)
    info = converter.model_info()
    print("Model info:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    converter.convert_to_onnx(Path(args.output_onnx))
    converter.convert_to_torchscript(Path(args.output_torchscript), method=args.method)

    if args.quantize:
        quantized_path = Path(args.output_torchscript).with_name("ecg_model_quantized.pt")
        quantized_model = converter.quantize_model()
        torch.jit.save(torch.jit.script(quantized_model), str(quantized_path))
        print(f"Saved quantized TorchScript model to {quantized_path}")

    print("Conversion finished.")


if __name__ == "__main__":
    main()
    converter.convert_to_onnx(output_path=args.output_onnx)
    
    # Конвертация в TorchScript
    print("\n" + "="*50)
    converter.convert_to_torchscript(
        output_path=args.output_torchscript,
        method=args.method
    )
    
    # Квантизация (опционально)
    if args.quantize:
        print("\n" + "="*50)
        quantized_model = converter.quantize_model(method='dynamic')
        quantized_path = args.output_torchscript.replace('.pt', '_quantized.pt')
        torch.jit.save(torch.jit.script(quantized_model), quantized_path)
        print(f"Квантизованная модель сохранена в {quantized_path}")
    
    print("\n" + "="*50)
    print("Конвертация завершена!")
    print(f"\nДля использования в мобильном приложении:")
    print(f"  - Android/iOS (ONNX): используйте {args.output_onnx} с ONNX Runtime")
    print(f"  - Android (TorchScript): используйте {args.output_torchscript} с PyTorch Mobile")
    print(f"  - iOS (TorchScript): используйте {args.output_torchscript} с LibTorch")
    print(f"  - Кроссплатформенные: используйте {args.output_onnx}")


if __name__ == '__main__':
    main()
