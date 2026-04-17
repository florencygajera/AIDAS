from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from scripts.evaluate_predictions import evaluate


def mine_failures(gt_path: Path, pred_path: Path, output: Path, iou_threshold: float = 0.5) -> dict:
    report = evaluate(gt_path, pred_path, iou_threshold)
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    preds = json.loads(pred_path.read_text(encoding="utf-8"))
    gt_map = {item["image"]: item for item in gt}
    pred_map = {item["image"]: item for item in preds}
    hard_negatives = []
    missed_defects = []
    false_positives = []
    for image, item in gt_map.items():
        gt_objs = item.get("objects", [])
        pred_objs = pred_map.get(image, {}).get("objects", [])
        if not pred_objs and gt_objs:
            missed_defects.append({"image": image, "reason": "no_predictions", "labels": [obj["label"] for obj in gt_objs]})
        if pred_objs and not gt_objs:
            false_positives.append({"image": image, "reason": "negative_sample", "predictions": pred_objs})
        for obj in pred_objs:
            if obj.get("label") in {"rust", "crack", "leak"} and obj.get("confidence", 0) < 0.5:
                hard_negatives.append({"image": image, "prediction": obj, "tag": "low_confidence_critical"})
    payload = {
        "report": report,
        "missed_defects": missed_defects,
        "false_positives": false_positives,
        "hard_negatives": hard_negatives,
        "next_actions": [
            "Add missed defects back into training with correct labels.",
            "Verify hard negatives for dirt, shadow, and reflections.",
            "Retrain with oversampling for crack and hole classes.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine false negatives and hard negatives from predictions.")
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", default=Path("outputs/failure_mining.json"), type=Path)
    args = parser.parse_args()
    payload = mine_failures(args.ground_truth, args.predictions, args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

