from __future__ import annotations

import io
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from ecg_artifact_filter import ECGArtifactFilter, ECGArtifactFilterConfig


def _mask_to_segments(mask: np.ndarray) -> List[Tuple[int, int]]:
    m = np.asarray(mask, dtype=bool).reshape(-1)
    if m.size == 0:
        return []
    diff = np.diff(m.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0]
    if m[0]:
        starts = np.concatenate(([0], starts))
    if m[-1]:
        ends = np.concatenate((ends, [m.size - 1]))
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def _segments_payload(segments: List[Tuple[int, int]], fs: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for s, e in segments:
        out.append(
            {
                "start_sample": s,
                "end_sample": e,
                "start_sec": round(s / float(fs), 6),
                "end_sec": round(e / float(fs), 6),
                "duration_sec": round((e - s + 1) / float(fs), 6),
            }
        )
    return out


class AnalyzeRequest(BaseModel):
    values: List[float] = Field(..., description="One-channel ECG values")
    samplingRateHz: int = Field(125, ge=1, description="Sampling rate in Hz")


def _build_filter() -> ECGArtifactFilter:
    model_path = Path(os.getenv("MODEL_PATH", "artifacts/ecg_rf_artifact_model_labelstudio.joblib"))
    if not model_path.is_absolute():
        model_path = (Path.cwd() / model_path).resolve()

    cfg = ECGArtifactFilterConfig(
        fs=int(os.getenv("ECG_FS", "125")),
        prob_threshold=float(os.getenv("PROB_THRESHOLD", "0.3")),
        prob_smooth_window=int(os.getenv("PROB_SMOOTH_WINDOW", "101")),
        min_artifact_duration=int(os.getenv("MIN_ARTIFACT_DURATION", "50")),
        gap_tolerance=int(os.getenv("GAP_TOLERANCE", "10")),
        normalize_input=True,
        include_variance=True,
        use_baseline_deviation_rule=True,
        baseline_window=125,
        baseline_std_k=3.5,
        use_plateau_rule=False,
        use_flatline_rule=False,
        use_multi_scale=True,
        multi_scale_min_durations=(50, 125, 188),
    )

    if model_path.exists():
        return ECGArtifactFilter.from_joblib(str(model_path), config=cfg)
    # Fallback to heuristic mode if model is absent.
    return ECGArtifactFilter(model=None, config=cfg)


FILTER = _build_filter()

app = FastAPI(
    title="PulseM ECG Artifact API",
    description="Detects artifact/normal intervals in 1-channel ECG JSON.",
    version="1.0.0",
)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


def _run_inference(values: List[float], fs: int) -> Dict[str, Any]:
    x = np.asarray(values, dtype=float).reshape(-1)
    if x.size == 0:
        raise HTTPException(status_code=400, detail="values is empty")

    res = FILTER.infer(x)
    artifact_segments = res["artifact_segments"]
    normal_segments = _mask_to_segments(res["normal_mask"])

    return {
        "samplingRateHz": fs,
        "samples": int(x.size),
        "model": {
            "mode": "rf" if FILTER.model is not None else "heuristic",
            "config": asdict(FILTER.config),
        },
        "summary": {
            "artifact_fraction": round(float(np.mean(res["artifact_mask"])), 6),
            "artifact_segment_count": len(artifact_segments),
            "normal_segment_count": len(normal_segments),
        },
        "artifact_intervals": _segments_payload(artifact_segments, fs),
        "normal_intervals": _segments_payload(normal_segments, fs),
    }


@app.post("/v1/analyze")
def analyze(request: AnalyzeRequest) -> Dict[str, Any]:
    return _run_inference(request.values, int(request.samplingRateHz))


@app.post("/v1/analyze-file")
async def analyze_file(file: UploadFile = File(...)) -> Dict[str, Any]:
    content = await file.read()
    try:
        payload = json.load(io.BytesIO(content))
    except Exception as e:  # pragma: no cover
        raise HTTPException(status_code=400, detail=f"Invalid JSON file: {e}") from e

    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="JSON root must be object")

    values = payload.get("values")
    fs = payload.get("samplingRateHz", 125)
    if not isinstance(values, list):
        raise HTTPException(status_code=400, detail="JSON must contain list field 'values'")
    if not isinstance(fs, int) or fs <= 0:
        raise HTTPException(status_code=400, detail="samplingRateHz must be positive int")

    return _run_inference(values, fs)

