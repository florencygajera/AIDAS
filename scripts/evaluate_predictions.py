from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def iou(box_a: List[float], box_b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = max(ax2 - ax1, 0) * max(ay2 - ay1, 0)
    area_b = max(bx2 - bx1, 0) * max(by2 - by1, 0)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def evaluate(gt_path: Path, pred_path: Path, iou_threshold: float = 0.5) -> dict:
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    preds = json.loads(pred_path.read_text(encoding="utf-8"))
    gt_map = {item["image"]: item for item in gt}
    pred_map = {item["image"]: item for item in preds}
    classes = sorted({obj["label"] for item in gt for obj in item.get("objects", [])} | {obj["label"] for item in preds for obj in item.get("objects", [])})
    tp = Counter()
    fp = Counter()
    fn = Counter()
    matches = []
    for image, item in gt_map.items():
        gt_objs = item.get("objects", [])
        pred_objs = pred_map.get(image, {}).get("objects", [])
        used = set()
        for gt_obj in gt_objs:
            best = None
            best_idx = None
            best_iou = 0.0
            for idx, pred_obj in enumerate(pred_objs):
                if idx in used or pred_obj.get("label") != gt_obj.get("label"):
                    continue
                value = iou(gt_obj["bbox"], pred_obj["bbox"])
                if value > best_iou:
                    best_iou = value
                    best = pred_obj
                    best_idx = idx
            if best is not None and best_iou >= iou_threshold:
                tp[gt_obj["label"]] += 1
                used.add(best_idx)
                matches.append({"image": image, "label": gt_obj["label"], "iou": round(best_iou, 4)})
            else:
                fn[gt_obj["label"]] += 1
        for idx, pred_obj in enumerate(pred_objs):
            if idx not in used:
                fp[pred_obj.get("label", "unknown")] += 1
    per_class = {}
    for cls in classes:
        precision = tp[cls] / max(tp[cls] + fp[cls], 1)
        recall = tp[cls] / max(tp[cls] + fn[cls], 1)
        per_class[cls] = {
            "tp": tp[cls],
            "fp": fp[cls],
            "fn": fn[cls],
            "precision": round(precision, 4),
            "recall": round(recall, 4),
        }
    critical = {cls: per_class.get(cls, {}) for cls in ("crack", "hole", "leak")}
    noncritical_fp = {cls: per_class.get(cls, {}) for cls in ("rust", "scratch")}
    overall_precision = sum(tp.values()) / max(sum(tp.values()) + sum(fp.values()), 1)
    overall_recall = sum(tp.values()) / max(sum(tp.values()) + sum(fn.values()), 1)
    return {
        "overall": {
            "precision": round(overall_precision, 4),
            "recall": round(overall_recall, 4),
        },
        "per_class": per_class,
        "critical_defects": critical,
        "false_positive_focus": noncritical_fp,
        "matches": matches,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate defect predictions against YOLO-style JSON.")
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", default=Path("outputs/evaluation.json"), type=Path)
    parser.add_argument("--iou", default=0.5, type=float)
    args = parser.parse_args()
    report = evaluate(args.ground_truth, args.predictions, args.iou)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

