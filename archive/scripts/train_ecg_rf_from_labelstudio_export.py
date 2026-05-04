"""
Train ECG RandomForest directly from Label Studio JSON export.

Expected export format: list of tasks, each task contains:
  - data.csv: local-files URL to CSV, e.g. /data/local-files/?d=artifacts/label_prep/file.csv
  - annotations[].result[] with type=timeserieslabels and labels Green/Red

Green -> normal (0), Red -> artifact (1)
Only explicitly labeled intervals are used for training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple
from urllib.parse import parse_qs, urlparse, unquote

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ecg_artifact_filter import ECGArtifactFilterConfig, _build_ecg_features


def local_file_url_to_path(url: str) -> Path:
    """
    Convert Label Studio local-files URL to an absolute file path.
    """
    parsed = urlparse(url)

    # Case 1: Local files URL -> /data/local-files/?d=relative/path.csv
    q = parse_qs(parsed.query)
    rel = q.get("d", [""])[0]
    if rel:
        rel = unquote(rel).replace("/", "\\")
        rel = rel.lstrip("\\")
        return (PROJECT_ROOT / rel).resolve()

    # Case 2: Uploaded into Label Studio -> /data/upload/<project>/<file.csv>
    # Try default Label Studio media dir on Windows.
    path = unquote(parsed.path or "")
    upload_marker = "/data/upload/"
    if upload_marker in path:
        sub = path.split(upload_marker, 1)[1].replace("/", "\\")
        candidates = [
            Path.home()
            / "AppData"
            / "Local"
            / "label-studio"
            / "label-studio"
            / "media"
            / "upload"
            / sub,
            Path.home()
            / "AppData"
            / "Local"
            / "label-studio"
            / "media"
            / "upload"
            / sub,
        ]
        for c in candidates:
            if c.exists():
                return c.resolve()

    return PROJECT_ROOT / "__labelstudio_unresolved_path__"


def task_intervals_to_targets(task: Dict[str, Any], t: np.ndarray) -> np.ndarray:
    """
    Build per-sample labels from one task annotation.

    Returns:
      y: int array with values {-1, 0, 1}
        -1 = unlabeled
         0 = Green/normal
         1 = Red/artifact
    """
    y = np.full(t.shape[0], -1, dtype=np.int8)
    annotations = task.get("annotations", [])
    if not annotations:
        return y

    # Use latest annotation if multiple exist.
    ann = annotations[-1]
    results = ann.get("result", [])
    for r in results:
        if r.get("type") != "timeserieslabels":
            continue
        v = r.get("value", {})
        start = v.get("start")
        end = v.get("end")
        labels = v.get("timeserieslabels", [])
        if start is None or end is None or end <= start:
            continue
        if not labels:
            continue

        label = str(labels[0]).strip().lower()
        val = 1 if label == "red" else 0 if label == "green" else None
        if val is None:
            continue

        idx = (t >= float(start)) & (t < float(end))
        # Red has priority if intervals overlap.
        if val == 1:
            y[idx] = 1
        else:
            y[(idx) & (y != 1)] = 0

    return y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--export-json",
        required=True,
        help="Path to Label Studio export JSON file",
    )
    parser.add_argument(
        "--model-out",
        default="artifacts/ecg_rf_artifact_model_labelstudio.joblib",
        help="Output model path",
    )
    args = parser.parse_args()

    export_path = Path(args.export_json)
    tasks = json.loads(export_path.read_text(encoding="utf-8"))
    if not isinstance(tasks, list) or not tasks:
        raise RuntimeError("Label Studio export must be a non-empty JSON array of tasks")

    cfg = ECGArtifactFilterConfig()
    all_x: List[np.ndarray] = []
    all_y: List[np.ndarray] = []

    for i, task in enumerate(tasks, start=1):
        data = task.get("data", {})
        csv_url = data.get("csv")
        if not csv_url:
            print(f"[task {i}] skip: no data.csv field")
            continue

        csv_path = local_file_url_to_path(str(csv_url))
        if not csv_path.exists():
            print(f"[task {i}] skip: csv not found -> {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        if "time" not in df.columns:
            raise RuntimeError(f"CSV has no 'time' column: {csv_path}")
        if "ecg_raw" not in df.columns:
            raise RuntimeError(f"CSV has no 'ecg_raw' column: {csv_path}")

        t = df["time"].to_numpy(dtype=float)
        sig = df["ecg_raw"].to_numpy(dtype=float)
        y = task_intervals_to_targets(task, t)
        labeled = y >= 0
        if not labeled.any():
            print(f"[task {i}] skip: no labeled intervals -> {csv_path.name}")
            continue

        # Robust normalization before feature extraction.
        x = sig - np.median(sig)
        s = float(np.std(x))
        if s > 1e-6:
            x = x / s

        feats = _build_ecg_features(
            x,
            cfg.window_sizes,
            include_variance=getattr(cfg, "include_variance", True),
        )
        all_x.append(feats[labeled])
        all_y.append(y[labeled].astype(int))

        frac_art = float((y[labeled] == 1).mean())
        print(
            f"[task {i}] {csv_path.name}: labeled={labeled.sum()}, "
            f"artifact_fraction={frac_art:.3f}"
        )

    if not all_x:
        raise RuntimeError("No labeled samples found in export")

    X = np.vstack(all_x)
    y = np.concatenate(all_y)
    classes = np.unique(y)
    if classes.size < 2:
        raise RuntimeError(
            f"Need both classes (Green and Red) to train RF, got classes={classes.tolist()}"
        )

    print(f"Total labeled samples: {X.shape[0]}, artifact_fraction={y.mean():.3f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=["Green", "Red"]))

    model_out = Path(args.model_out)
    if not model_out.is_absolute():
        model_out = PROJECT_ROOT / model_out
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_out)
    print(f"Saved model to: {model_out}")


if __name__ == "__main__":
    main()

