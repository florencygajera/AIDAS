from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from app.config import (
    CLASS_NAMES,
    IMAGE_SIZE,
    MERGE_CONFIDENCE,
    PATCH_SIZE,
    PATCH_STRIDE,
    PRIMARY_CONFIDENCE,
    YOLO_PRIMARY_WEIGHTS,
    YOLO_SEGMENT_WEIGHTS,
)
from app.services.preprocess import PreprocessedImage


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    source: str = "primary"
    source_tags: Tuple[str, ...] = ()
    mask_area: Optional[float] = None
    frame_index: Optional[int] = None
    timestamp: Optional[float] = None
    source_count: int = 1
    refined_confidence: Optional[float] = None

    @property
    def effective_confidence(self) -> float:
        return float(self.refined_confidence if self.refined_confidence is not None else self.confidence)


def _iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter_area = float((inter_x2 - inter_x1) * (inter_y2 - inter_y1))
    area_a = float(max(ax2 - ax1, 0) * max(ay2 - ay1, 0))
    area_b = float(max(bx2 - bx1, 0) * max(by2 - by1, 0))
    denom = area_a + area_b - inter_area
    return inter_area / denom if denom > 0 else 0.0


def _clamp_box(box: Tuple[float, float, float, float], size: int = IMAGE_SIZE) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x1 = int(max(0, min(size - 1, round(x1))))
    y1 = int(max(0, min(size - 1, round(y1))))
    x2 = int(max(0, min(size - 1, round(x2))))
    y2 = int(max(0, min(size - 1, round(y2))))
    if x2 <= x1:
        x2 = min(size - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(size - 1, y1 + 1)
    return x1, y1, x2, y2


def _xyxy_from_xywh(center_x: float, center_y: float, width: float, height: float) -> Tuple[float, float, float, float]:
    return center_x - width / 2.0, center_y - height / 2.0, center_x + width / 2.0, center_y + height / 2.0


class YoloDetector:
    def __init__(self, weights: str = YOLO_PRIMARY_WEIGHTS, conf: float = PRIMARY_CONFIDENCE, iou: float = 0.45):
        self.weights = weights
        self.conf = conf
        self.iou = iou
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        try:
            from ultralytics import YOLO
        except Exception as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "ultralytics is required for detection inference. Install project dependencies first."
            ) from exc
        self._model = YOLO(self.weights)
        return self._model

    def predict(self, image_rgb: np.ndarray, source: str = "primary") -> List[Detection]:
        model = self._ensure_model()
        results = model.predict(source=image_rgb, conf=self.conf, iou=self.iou, verbose=False)
        detections: List[Detection] = []
        if not results:
            return detections
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return detections
        names = result.names if hasattr(result, "names") else {}
        for box in boxes:
            cls_idx = int(box.cls.item())
            label = str(names.get(cls_idx, CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else f"class_{cls_idx}"))
            confidence = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                Detection(
                    label=label,
                    confidence=confidence,
                    bbox=_clamp_box((x1, y1, x2, y2)),
                    source=source,
                    source_tags=(source,),
                )
            )
        masks = getattr(result, "masks", None)
        if masks is not None and getattr(masks, "xy", None):
            for idx, segment in enumerate(masks.xy):
                if idx >= len(detections):
                    break
                polygon = np.asarray(segment, dtype=np.float32)
                if polygon.size == 0:
                    continue
                area = float(cv2.contourArea(polygon.astype(np.float32)))
                detections[idx].mask_area = area
        return detections


class OptionalSegmenter(YoloDetector):
    def __init__(self, weights: str = YOLO_SEGMENT_WEIGHTS):
        super().__init__(weights=weights, conf=PRIMARY_CONFIDENCE, iou=0.45)

    @property
    def available(self) -> bool:
        return bool(self.weights)


def denormalize_patch_box(
    patch_bbox: Tuple[int, int, int, int],
    patch_origin: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    ox, oy = patch_origin
    x1, y1, x2, y2 = patch_bbox
    return x1 + ox, y1 + oy, x2 + ox, y2 + oy


def sliding_windows(image_size: int = IMAGE_SIZE, patch_size: int = PATCH_SIZE, stride: int = PATCH_STRIDE):
    for y in range(0, max(image_size - patch_size + 1, 1), stride):
        for x in range(0, max(image_size - patch_size + 1, 1), stride):
            yield x, y, min(x + patch_size, image_size), min(y + patch_size, image_size)


def run_patch_detection(detector: YoloDetector, image_rgb: np.ndarray) -> List[Detection]:
    detections: List[Detection] = []
    for x1, y1, x2, y2 in sliding_windows():
        patch = image_rgb[y1:y2, x1:x2]
        if patch.size == 0:
            continue
        patch_detections = detector.predict(patch, source="patch")
        for det in patch_detections:
            px1, py1, px2, py2 = det.bbox
            det.bbox = _clamp_box((px1 + x1, py1 + y1, px2 + x1, py2 + y1))
            detections.append(det)
    return detections


def run_multiscale_detection(detector: YoloDetector, image_rgb: np.ndarray) -> List[Detection]:
    full_scale = detector.predict(image_rgb, source="full_640")
    downscaled = cv2.resize(image_rgb, (320, 320), interpolation=cv2.INTER_AREA)
    down_scale_detections = detector.predict(downscaled, source="downscale_320")
    merged = full_scale + down_scale_detections + run_patch_detection(detector, image_rgb)
    return merged


def weighted_box_fusion(detections: Sequence[Detection], iou_threshold: float = 0.5) -> List[Detection]:
    grouped: Dict[str, List[Detection]] = {}
    for det in detections:
        grouped.setdefault(det.label, []).append(det)
    fused: List[Detection] = []
    for label, items in grouped.items():
        items = sorted(items, key=lambda d: d.effective_confidence, reverse=True)
        used = [False] * len(items)
        for idx, item in enumerate(items):
            if used[idx]:
                continue
            cluster = [item]
            used[idx] = True
            for jdx in range(idx + 1, len(items)):
                if used[jdx]:
                    continue
                if _iou(item.bbox, items[jdx].bbox) >= iou_threshold:
                    cluster.append(items[jdx])
                    used[jdx] = True
            if len(cluster) == 1:
                fused.append(item)
                continue
            weights = np.array([max(det.effective_confidence, 1e-3) for det in cluster], dtype=np.float32)
            coords = np.array([det.bbox for det in cluster], dtype=np.float32)
            averaged = np.sum(coords * weights[:, None], axis=0) / np.sum(weights)
            confidence = float(max(det.effective_confidence for det in cluster))
            source_count = sum(det.source_count for det in cluster)
            source_tags = tuple(sorted({tag for det in cluster for tag in det.source_tags} | {det.source for det in cluster}))
            fused.append(
                Detection(
                    label=label,
                    confidence=confidence,
                    bbox=_clamp_box(tuple(averaged.tolist())),
                    source="fused",
                    source_tags=source_tags,
                    mask_area=max((det.mask_area or 0.0) for det in cluster) or None,
                    source_count=source_count,
                )
            )
    return sorted(fused, key=lambda d: d.effective_confidence, reverse=True)


def non_max_suppression(detections: Sequence[Detection], iou_threshold: float = 0.45) -> List[Detection]:
    ordered = sorted(detections, key=lambda d: d.effective_confidence, reverse=True)
    kept: List[Detection] = []
    for det in ordered:
        keep = True
        for existing in kept:
            if det.label == existing.label and _iou(det.bbox, existing.bbox) >= iou_threshold:
                keep = False
                break
        if keep:
            kept.append(det)
    return kept


def remap_to_original(preprocessed: PreprocessedImage, det: Detection) -> Detection:
    x1, y1, x2, y2 = det.bbox
    scale = preprocessed.scale
    pad_x = preprocessed.pad_x
    pad_y = preprocessed.pad_y
    ox1 = int(round((x1 - pad_x) / scale))
    oy1 = int(round((y1 - pad_y) / scale))
    ox2 = int(round((x2 - pad_x) / scale))
    oy2 = int(round((y2 - pad_y) / scale))
    h, w = preprocessed.original.shape[:2]
    ox1 = max(0, min(w - 1, ox1))
    oy1 = max(0, min(h - 1, oy1))
    ox2 = max(0, min(w - 1, ox2))
    oy2 = max(0, min(h - 1, oy2))
    if ox2 <= ox1:
        ox2 = min(w - 1, ox1 + 1)
    if oy2 <= oy1:
        oy2 = min(h - 1, oy1 + 1)
    det.bbox = (ox1, oy1, ox2, oy2)
    return det
