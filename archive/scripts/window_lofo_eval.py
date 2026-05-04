"""
Window-level RF + strict leave-one-file-out evaluation.
Uses features from raw + bandpass-filtered ECG.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def resolve_csv_path(csv_url: str) -> Path:
    parsed = urlparse(csv_url)
    q = parse_qs(parsed.query)
    rel = q.get("d", [""])[0]
    if rel:
        rel = unquote(rel).replace("/", "\\").lstrip("\\")
        p = (PROJECT_ROOT / rel).resolve()
        if p.exists():
            return p
    path = unquote(parsed.path or "")
    marker = "/data/upload/"
    if marker in path:
        sub = path.split(marker, 1)[1].replace("/", "\\")
        for c in [
            Path.home() / "AppData" / "Local" / "label-studio" / "label-studio" / "media" / "upload" / sub,
            Path.home() / "AppData" / "Local" / "label-studio" / "media" / "upload" / sub,
        ]:
            if c.exists():
                return c.resolve()
    return Path()


def labels_from_task(task: Dict[str, Any], t: np.ndarray) -> np.ndarray:
    y = np.full(t.shape[0], -1, dtype=np.int8)
    anns = task.get("annotations", [])
    if not anns:
        return y
    for r in anns[-1].get("result", []):
        if r.get("type") != "timeserieslabels":
            continue
        v = r.get("value", {})
        labs = v.get("timeserieslabels", [])
        if not labs:
            continue
        s = v.get("start")
        e = v.get("end")
        if s is None or e is None or e <= s:
            continue
        val = 1 if str(labs[0]).lower() == "red" else 0 if str(labs[0]).lower() == "green" else None
        if val is None:
            continue
        m = (t >= float(s)) & (t < float(e))
        if val == 1:
            y[m] = 1
        else:
            y[(m) & (y != 1)] = 0
    return y


def bandpass(x: np.ndarray, fs: int = 125, low: float = 0.5, high: float = 20.0, order: int = 4) -> np.ndarray:
    ny = fs / 2.0
    b, a = butter(order, [max(low / ny, 1e-5), min(high / ny, 0.999)], btype="bandpass")
    return filtfilt(b, a, x)


def robust_norm(x: np.ndarray) -> np.ndarray:
    z = x - np.median(x)
    s = float(np.std(z))
    return z / s if s > 1e-6 else z


def featurize_windows(x_raw: np.ndarray, x_filt: np.ndarray, y: np.ndarray, w: int, step: int) -> Tuple[np.ndarray, np.ndarray]:
    feats, labels = [], []
    for s in range(0, len(x_raw) - w + 1, step):
        e = s + w
        yw = y[s:e]
        lab = yw >= 0
        if not lab.any():
            continue
        red_ratio = float((yw[lab] == 1).mean())
        # Window label: red if >=20% red samples in labeled part.
        wl = 1 if red_ratio >= 0.2 else 0

        rw = x_raw[s:e]
        fw = x_filt[s:e]
        dr = np.diff(rw, prepend=rw[0])
        df = np.diff(fw, prepend=fw[0])
        f = [
            float(np.mean(rw)), float(np.std(rw)), float(np.var(rw)), float(np.mean(np.abs(dr))),
            float(np.mean(fw)), float(np.std(fw)), float(np.var(fw)), float(np.mean(np.abs(df))),
            float(np.percentile(np.abs(rw), 95)), float(np.percentile(np.abs(fw), 95)),
        ]
        feats.append(f)
        labels.append(wl)
    return np.asarray(feats, dtype=float), np.asarray(labels, dtype=int)


def prf(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-json", required=True)
    ap.add_argument("--window", type=int, default=188)
    ap.add_argument("--step", type=int, default=62)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tasks = json.loads(Path(args.export_json).read_text(encoding="utf-8"))
    recs: List[Tuple[np.ndarray, np.ndarray, str]] = []

    for t in tasks:
        csv_url = t.get("data", {}).get("csv")
        if not csv_url:
            continue
        p = resolve_csv_path(str(csv_url))
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if "time" not in df.columns or "ecg_raw" not in df.columns:
            continue
        tt = df["time"].to_numpy(float)
        xx = df["ecg_raw"].to_numpy(float)
        yy = labels_from_task(t, tt)
        if not (yy >= 0).any():
            continue
        x_raw = robust_norm(xx)
        x_flt = robust_norm(bandpass(xx))
        Xw, yw = featurize_windows(x_raw, x_flt, yy, args.window, args.step)
        if yw.size == 0 or np.unique(yw).size < 2:
            continue
        recs.append((Xw, yw, p.name))
        print(f"{p.name}: windows={len(yw)}, red_frac={float((yw==1).mean()):.3f}")

    if len(recs) < 3:
        raise RuntimeError("Not enough labeled files with both classes at window level.")

    all_t, all_p = [], []
    for test_i in range(len(recs)):
        remain = [i for i in range(len(recs)) if i != test_i]
        val_i = remain[0]
        train_i = remain[1:] if len(remain) > 1 else remain

        Xtr = np.vstack([recs[i][0] for i in train_i])
        ytr = np.concatenate([recs[i][1] for i in train_i])
        if np.unique(ytr).size < 2:
            continue

        clf = RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced",
            n_jobs=-1,
            random_state=args.seed,
        )
        clf.fit(Xtr, ytr)

        Xv, yv, _ = recs[val_i]
        pv = clf.predict_proba(Xv)[:, 1]
        best_th, best_f1 = 0.5, -1.0
        for th in [0.3, 0.4, 0.5, 0.6]:
            f1 = prf(yv, (pv >= th).astype(int))[2]
            if f1 > best_f1:
                best_f1, best_th = f1, th

        Xt, yt, name = recs[test_i]
        yp = (clf.predict_proba(Xt)[:, 1] >= best_th).astype(int)
        p, r, f1 = prf(yt, yp)
        print(f"[test {name}] th={best_th:.2f} P={p:.3f} R={r:.3f} F1={f1:.3f}")
        all_t.append(yt)
        all_p.append(yp)

    yt = np.concatenate(all_t)
    yp = np.concatenate(all_p)
    p, r, f1 = prf(yt, yp)
    print(f"\nLOFO window-level overall: P={p:.3f} R={r:.3f} F1={f1:.3f}, n={len(yt)}")


if __name__ == "__main__":
    main()

