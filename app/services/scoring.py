from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from app.config import CRITICAL_CLASSES, SEVERITY_WEIGHTS
from app.services.detection import Detection


def severity_from_label(label: str) -> str:
    if label in {"crack", "hole", "leak"}:
        return "HIGH"
    if label in {"dent", "corrosion", "rust"}:
        return "MEDIUM"
    return "LOW"


def _area_ratio(bbox: Tuple[int, int, int, int], image_size: Tuple[int, int]) -> float:
    x1, y1, x2, y2 = bbox
    width, height = image_size
    area = max(x2 - x1, 0) * max(y2 - y1, 0)
    return float(area) / float(max(width * height, 1))


def risk_score_for_detections(
    detections: Sequence[Detection],
    image_size: Tuple[int, int],
    consistency_counts: Optional[Dict[Tuple[str, Tuple[int, int, int, int]], int]] = None,
) -> float:
    score = 0.0
    for det in detections:
        weight = SEVERITY_WEIGHTS.get(det.label, 1.0)
        consistency = 1.0
        if consistency_counts is not None:
            consistency = min(1.0, consistency_counts.get((det.label, det.bbox), 1) / 3.0)
            consistency = max(consistency, 0.33)
        area = _area_ratio(det.bbox, image_size)
        confidence = max(det.effective_confidence, 0.0)
        score += weight * area * confidence * consistency
    return float(round(score, 6))


def audit_decision(detections: Sequence[Detection], risk_score: float) -> str:
    if any(det.label in CRITICAL_CLASSES for det in detections):
        return "REJECTED"
    if risk_score > 1.0:
        return "NOT_READY"
    if 0.3 <= risk_score <= 1.0:
        return "REVIEW"
    return "READY"

