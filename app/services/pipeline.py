from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from app.config import (
    CRITICAL_CLASSES,
    FINAL_VALIDATION_CONFIDENCE,
    IMAGE_SIZE,
    ENABLE_PATCH_SCAN,
    NMS_IOU,
    MERGE_CONFIDENCE,
    OUTPUT_DIR,
    SECONDARY_CONFIRMATION_CONFIDENCE,
    TEMPORAL_MIN_CONSEC_FRAMES,
    VIDEO_MIN_FRAMES,
    VIDEO_SAMPLE_INTERVAL_SECONDS,
)
from app.schemas import AuditResponse, DefectBox
from app.services.calibration import ConfidenceCalibrator
from app.services.classification import DefectRefiner
from app.services.detection import (
    Detection,
    OptionalSegmenter,
    YoloDetector,
    non_max_suppression,
    remap_to_original,
    run_multiscale_detection,
    run_patch_detection,
    weighted_box_fusion,
)
from app.services.preprocess import (
    is_low_quality,
    is_supported_media,
    is_video,
    load_video_frames,
    preprocess_image,
    read_image_rgb,
    save_rgb_image,
)
from app.services.temporal import TemporalConsistencyTracker
from app.services.validation import validate_detection
from app.services.scoring import audit_decision, risk_score_for_detections, severity_from_label


def _detection_to_box(det: Detection) -> DefectBox:
    return DefectBox(
        type=det.label,
        confidence=round(float(det.effective_confidence), 4),
        bbox=[int(v) for v in det.bbox],
        severity=severity_from_label(det.label),
    )


def _output_url(path: Path) -> str:
    return f"/outputs/{path.name}"


def _draw_annotations(image_rgb: np.ndarray, detections: Sequence[Detection]) -> np.ndarray:
    annotated = image_rgb.copy()
    colors = {
        "HIGH": (220, 38, 38),
        "MEDIUM": (245, 158, 11),
        "LOW": (34, 197, 94),
    }
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        severity = severity_from_label(det.label)
        color = colors[severity]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{det.label} {det.effective_confidence:.2f}"
        if det.mask_area is not None:
            label += f" area:{det.mask_area:.0f}"
        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(y1 - text_size[1] - 8, 0)
        cv2.rectangle(annotated, (x1, top), (x1 + text_size[0] + 6, top + text_size[1] + baseline + 6), color, -1)
        cv2.putText(
            annotated,
            label,
            (x1 + 3, top + text_size[1] + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return annotated


def _merge_video_detections(detections: Sequence[Detection]) -> Tuple[List[Detection], Dict[Tuple[str, Tuple[int, int, int, int]], int]]:
    if not detections:
        return [], {}
    groups: List[List[Detection]] = []
    for det in detections:
        placed = False
        for group in groups:
            if group[0].label != det.label:
                continue
            if any(_iou(group_item.bbox, det.bbox) >= 0.4 for group_item in group):
                group.append(det)
                placed = True
                break
        if not placed:
            groups.append([det])
    merged: List[Detection] = []
    consistency_counts: Dict[Tuple[str, Tuple[int, int, int, int]], int] = {}
    for group in groups:
        fused = weighted_box_fusion(group, iou_threshold=0.4)
        for det in fused:
            count = len(group)
            det.source_count = count
            consistency_counts[(det.label, det.bbox)] = count
            merged.append(det)
    return merged, consistency_counts


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


class DefectPipeline:
    def __init__(
        self,
        detector: Optional[YoloDetector] = None,
        segmenter: Optional[OptionalSegmenter] = None,
        refiner: Optional[DefectRefiner] = None,
        calibrator: Optional[ConfidenceCalibrator] = None,
    ):
        self.detector = detector or YoloDetector()
        self.segmenter = segmenter or OptionalSegmenter()
        self.refiner = refiner or DefectRefiner()
        self.calibrator = calibrator or ConfidenceCalibrator()

    def _detect_single_image(self, image_rgb: np.ndarray) -> Tuple[List[Detection], Dict[Tuple[str, Tuple[int, int, int, int]], int]]:
        detections = run_multiscale_detection(self.detector, image_rgb, include_patch_scan=False)
        if self.segmenter and self.segmenter.available:
            detections.extend(self.segmenter.predict(image_rgb, source="segment"))
        if not detections and ENABLE_PATCH_SCAN:
            detections = run_multiscale_detection(self.detector, image_rgb, include_patch_scan=True)
            if self.segmenter and self.segmenter.available:
                detections.extend(self.segmenter.predict(image_rgb, source="segment"))
        fused = weighted_box_fusion(detections, iou_threshold=NMS_IOU)
        fused = non_max_suppression(fused, iou_threshold=NMS_IOU)
        refined: List[Detection] = []
        image_size = (image_rgb.shape[1], image_rgb.shape[0])
        for det in fused:
            det.refined_confidence = self.calibrator.calibrate_probability(det.confidence)
            det = self.refiner.refine_detection(image_rgb, det)
            validation = validate_detection(det, image_size, CRITICAL_CLASSES)
            if not validation.accepted and det.label not in CRITICAL_CLASSES:
                continue
            det.source_count = max(det.source_count, len(set(det.source_tags)))
            if det.label in CRITICAL_CLASSES:
                if det.source_count >= 2 or det.effective_confidence >= SECONDARY_CONFIRMATION_CONFIDENCE:
                    refined.append(det)
                elif validation.accepted and det.effective_confidence >= FINAL_VALIDATION_CONFIDENCE:
                    refined.append(det)
            elif det.effective_confidence >= FINAL_VALIDATION_CONFIDENCE:
                refined.append(det)
        refined = non_max_suppression(refined, iou_threshold=NMS_IOU)
        consistency = {(det.label, det.bbox): max(det.source_count, len(set(det.source_tags))) for det in refined}
        return refined, consistency

    def process_image(self, input_path: Path) -> AuditResponse:
        image_rgb = read_image_rgb(input_path)
        preprocessed = preprocess_image(image_rgb)
        detections, consistency = self._detect_single_image(preprocessed.image)
        remapped: List[Detection] = []
        for det in detections:
            remapped.append(remap_to_original(preprocessed, det))
        risk = risk_score_for_detections(remapped, (image_rgb.shape[1], image_rgb.shape[0]), consistency)
        status = audit_decision(remapped, risk)
        annotated = _draw_annotations(image_rgb, remapped)
        out_path = OUTPUT_DIR / f"{input_path.stem}_annotated.jpg"
        save_rgb_image(annotated, out_path)
        return AuditResponse(
            status=status,
            risk_score=risk,
            defects=[_detection_to_box(det) for det in remapped],
            annotated_image=_output_url(out_path),
            annotated_video=None,
            source_type="image",
            frame_count=1,
            skipped_frames=0,
        )

    def process_video(self, input_path: Path) -> AuditResponse:
        detections_all: List[Detection] = []
        frame_summaries: List[Tuple[int, np.ndarray, List[Detection]]] = []
        skipped_frames = 0
        frame_count = 0
        sampled_index = 0
        first_frame_rgb: Optional[np.ndarray] = None
        tracker = TemporalConsistencyTracker()
        for frame_idx, frame_rgb, timestamp in load_video_frames(input_path, VIDEO_SAMPLE_INTERVAL_SECONDS):
            frame_count += 1
            if first_frame_rgb is None:
                first_frame_rgb = frame_rgb
            if is_low_quality(frame_rgb):
                skipped_frames += 1
                continue
            preprocessed = preprocess_image(frame_rgb)
            detections, _ = self._detect_single_image(preprocessed.image)
            remapped = [remap_to_original(preprocessed, det) for det in detections]
            for det in remapped:
                det.frame_index = frame_idx
                det.timestamp = timestamp
            frame_summaries.append((frame_idx, frame_rgb, remapped))
            detections_all.extend(remapped)
            tracker.update(remapped, sampled_index)
            sampled_index += 1
        if not detections_all:
            fallback = first_frame_rgb if first_frame_rgb is not None else np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
            out_path = OUTPUT_DIR / f"{input_path.stem}_annotated.jpg"
            save_rgb_image(fallback, out_path)
            return AuditResponse(
                status="READY",
                risk_score=0.0,
                defects=[],
                annotated_image=_output_url(out_path),
                annotated_video=None,
                source_type="video",
                frame_count=frame_count,
                skipped_frames=skipped_frames,
            )
        confirmed = tracker.confirmed_detections(min_consecutive_frames=TEMPORAL_MIN_CONSEC_FRAMES)
        if not confirmed:
            confirmed = []
            for track in tracker.tracks:
                if track.total_hits >= VIDEO_MIN_FRAMES:
                    det = track.best_detection
                    det.source_count = track.total_hits
                    confirmed.append(det)
        confirmed = non_max_suppression(weighted_box_fusion(confirmed, iou_threshold=NMS_IOU), iou_threshold=NMS_IOU)
        refined = [self.refiner.refine_detection(first_frame_rgb if first_frame_rgb is not None else frame_summaries[0][1], det) for det in confirmed]
        consistency_counts = {(det.label, det.bbox): max(det.source_count, len(set(det.source_tags))) for det in refined}
        risk = risk_score_for_detections(
            refined,
            (frame_summaries[0][1].shape[1], frame_summaries[0][1].shape[0]) if frame_summaries else (IMAGE_SIZE, IMAGE_SIZE),
            consistency_counts,
        )
        status = audit_decision(refined, risk)
        annotated_source = first_frame_rgb if first_frame_rgb is not None else frame_summaries[0][1]
        annotated = _draw_annotations(annotated_source, refined)
        out_path = OUTPUT_DIR / f"{input_path.stem}_annotated.jpg"
        save_rgb_image(annotated, out_path)
        return AuditResponse(
            status=status,
            risk_score=risk,
            defects=[_detection_to_box(det) for det in refined],
            annotated_image=_output_url(out_path),
            annotated_video=None,
            source_type="video",
            frame_count=frame_count,
            skipped_frames=skipped_frames,
        )

    def process(self, input_path: Path) -> AuditResponse:
        if not is_supported_media(input_path):
            raise ValueError("Unsupported file type. Use JPG, PNG, JPEG, BMP, MP4, AVI, or MOV.")
        if is_video(input_path):
            return self.process_video(input_path)
        return self.process_image(input_path)
