from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from app.config import CLASSIFIER_THRESHOLD, CLASS_NAMES, CLASSIFIER_WEIGHTS
from app.services.detection import Detection


@dataclass
class ClassificationResult:
    label: str
    score: float


class DefectRefiner:
    def __init__(self, weights: str = CLASSIFIER_WEIGHTS, threshold: float = CLASSIFIER_THRESHOLD):
        self.weights = weights
        self.threshold = threshold
        self._model = None
        self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        if not self.weights:
            self._available = False
            return None
        weights_path = Path(self.weights)
        if not weights_path.exists():
            self._available = False
            return None
        try:
            import torch
            from torchvision import models
        except Exception:
            self._available = False
            return None
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))
        state = torch.load(weights_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        elif isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
            state = state["model"]
        if isinstance(state, dict):
            cleaned = {}
            for key, value in state.items():
                cleaned[key.removeprefix("module.") if hasattr(key, "removeprefix") else key.replace("module.", "", 1)] = value
            state = cleaned
            model.load_state_dict(state, strict=False)
        else:
            model.load_state_dict(state, strict=False)
        model.eval()
        self._model = model
        self._available = True
        return self._model

    def classify(self, image_rgb: np.ndarray) -> Optional[ClassificationResult]:
        model = self._ensure_model()
        if model is None:
            return None
        try:
            import torch
            from torchvision import transforms
        except Exception:
            self._available = False
            return None
        resized = cv2.resize(image_rgb, (224, 224), interpolation=cv2.INTER_AREA)
        tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )(resized).unsqueeze(0)
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            score, index = torch.max(probs, dim=0)
        return ClassificationResult(label=CLASS_NAMES[int(index.item())], score=float(score.item()))

    def refine_detection(self, image_rgb: np.ndarray, detection: Detection) -> Detection:
        if not self.available:
            detection.refined_confidence = detection.confidence
            return detection
        x1, y1, x2, y2 = detection.bbox
        crop = image_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            detection.refined_confidence = detection.confidence
            return detection
        result = self.classify(crop)
        if result is None:
            detection.refined_confidence = detection.confidence
            return detection
        if result.score >= self.threshold and result.label == detection.label:
            detection.refined_confidence = max(detection.confidence, result.score)
            detection.source_tags = tuple(sorted(set(detection.source_tags) | {"classifier"}))
        elif result.score >= self.threshold and result.label != detection.label:
            detection.refined_confidence = min(detection.confidence, result.score * 0.75)
        else:
            detection.refined_confidence = min(detection.confidence, result.score)
        return detection
