from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from app.config import (
    ASPECT_RATIO_MAX,
    ASPECT_RATIO_MIN,
    CRITICAL_ZONE_JSON,
    SMALL_AREA_RATIO,
)
from app.services.detection import Detection


@dataclass
class ValidationResult:
    accepted: bool
    reason: str = ""


def _area_ratio(bbox: Tuple[int, int, int, int], image_size: Tuple[int, int]) -> float:
    x1, y1, x2, y2 = bbox
    width, height = image_size
    area = max(x2 - x1, 0) * max(y2 - y1, 0)
    return float(area) / float(max(width * height, 1))


def aspect_ratio(bbox: Tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = bbox
    width = max(x2 - x1, 1)
    height = max(y2 - y1, 1)
    return float(max(width, height)) / float(min(width, height))


def load_critical_zone() -> Optional[dict]:
    if not CRITICAL_ZONE_JSON:
        return None
    path = Path(CRITICAL_ZONE_JSON)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    try:
        return json.loads(CRITICAL_ZONE_JSON)
    except Exception:
        return None


def location_factor(bbox: Tuple[int, int, int, int], image_size: Tuple[int, int], critical_zone: Optional[dict] = None) -> float:
    width, height = image_size
    x1, y1, x2, y2 = bbox
    cx = ((x1 + x2) / 2.0) / max(width, 1)
    cy = ((y1 + y2) / 2.0) / max(height, 1)
    if critical_zone:
        try:
            x_min, y_min, x_max, y_max = critical_zone["x_min"], critical_zone["y_min"], critical_zone["x_max"], critical_zone["y_max"]
            if x_min <= cx <= x_max and y_min <= cy <= y_max:
                return 1.35
        except Exception:
            pass
    center_distance = abs(cx - 0.5) + abs(cy - 0.5)
    if center_distance < 0.35:
        return 1.15
    if cx < 0.1 or cx > 0.9 or cy < 0.1 or cy > 0.9:
        return 0.90
    return 1.0


def validate_detection(
    detection: Detection,
    image_size: Tuple[int, int],
    critical_classes: Sequence[str],
    min_area_ratio: float = SMALL_AREA_RATIO,
) -> ValidationResult:
    area = _area_ratio(detection.bbox, image_size)
    ar = aspect_ratio(detection.bbox)
    if area < min_area_ratio and detection.label not in critical_classes:
        return ValidationResult(False, "tiny_area")
    if not (ASPECT_RATIO_MIN <= ar <= ASPECT_RATIO_MAX):
        if detection.label in critical_classes and area >= min_area_ratio * 0.5:
            return ValidationResult(True, "critical_shape_override")
        return ValidationResult(False, "bad_aspect_ratio")
    return ValidationResult(True, "ok")

