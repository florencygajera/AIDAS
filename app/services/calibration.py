from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from app.config import CALIBRATION_WEIGHTS, FINAL_VALIDATION_CONFIDENCE


@dataclass
class CalibrationProfile:
    temperature: float = 1.0
    class_thresholds: Optional[Dict[str, float]] = None


class ConfidenceCalibrator:
    def __init__(self, weights: str = CALIBRATION_WEIGHTS):
        self.weights = weights
        self.profile = CalibrationProfile()
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        if not self.weights:
            self._loaded = True
            return
        path = Path(self.weights)
        if not path.exists():
            self._loaded = True
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            self._loaded = True
            return
        temperature = float(payload.get("temperature", 1.0))
        thresholds = payload.get("class_thresholds") or {}
        self.profile = CalibrationProfile(
            temperature=max(temperature, 1e-3),
            class_thresholds={str(k): float(v) for k, v in thresholds.items()} if isinstance(thresholds, dict) else None,
        )
        self._loaded = True

    def calibrate_probability(self, probability: float) -> float:
        self._load()
        p = float(min(max(probability, 1e-6), 1.0 - 1e-6))
        if self.profile.temperature <= 1.0:
            return p
        logit = math.log(p / (1.0 - p))
        scaled = logit / self.profile.temperature
        calibrated = 1.0 / (1.0 + math.exp(-scaled))
        return float(min(max(calibrated, 0.0), 1.0))

    def threshold_for_class(self, label: str) -> float:
        self._load()
        if self.profile.class_thresholds and label in self.profile.class_thresholds:
            return float(self.profile.class_thresholds[label])
        return FINAL_VALIDATION_CONFIDENCE

