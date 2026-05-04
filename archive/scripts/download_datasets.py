"""
CLI helper that downloads and preprocesses MIT-BIH datasets.
"""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from pulsem.datasets.mit_bih import prepare_mit_bih_datasets


def main():
    project_root = Path(__file__).resolve().parents[1]
    raw_root = project_root / "data" / "raw"
    processed_root = project_root / "data" / "processed"

    dataset_path = prepare_mit_bih_datasets(raw_root, processed_root)
    print(f"Saved processed dataset to {dataset_path}")


if __name__ == "__main__":
    main()

