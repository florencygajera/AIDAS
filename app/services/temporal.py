from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from app.services.detection import Detection


def iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
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


@dataclass
class TemporalTrack:
    label: str
    bbox: Tuple[int, int, int, int]
    detections: List[Detection] = field(default_factory=list)
    last_frame_index: int = -1
    consecutive_hits: int = 0
    total_hits: int = 0

    def update(self, detection: Detection, frame_index: int) -> None:
        self.detections.append(detection)
        self.bbox = detection.bbox
        self.total_hits += 1
        if self.last_frame_index >= 0 and frame_index == self.last_frame_index + 1:
            self.consecutive_hits += 1
        else:
            self.consecutive_hits = 1
        self.last_frame_index = frame_index

    @property
    def best_detection(self) -> Detection:
        return max(self.detections, key=lambda det: det.effective_confidence)


class TemporalConsistencyTracker:
    def __init__(self, iou_threshold: float = 0.4):
        self.iou_threshold = iou_threshold
        self.tracks: List[TemporalTrack] = []

    def update(self, detections: Sequence[Detection], frame_index: int) -> None:
        for det in detections:
            matched = None
            for track in self.tracks:
                if track.label != det.label:
                    continue
                if iou(track.bbox, det.bbox) >= self.iou_threshold:
                    matched = track
                    break
            if matched is None:
                track = TemporalTrack(label=det.label, bbox=det.bbox)
                track.update(det, frame_index)
                self.tracks.append(track)
            else:
                matched.update(det, frame_index)

    def confirmed_detections(self, min_consecutive_frames: int = 2) -> List[Detection]:
        confirmed: List[Detection] = []
        for track in self.tracks:
            if track.consecutive_hits >= min_consecutive_frames:
                det = track.best_detection
                det.source_count = max(det.source_count, track.consecutive_hits)
                confirmed.append(det)
        return confirmed

