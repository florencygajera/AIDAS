from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List


def load_yolo_labels(label_dir: Path) -> Dict[str, List[str]]:
    labels: Dict[str, List[str]] = {}
    for path in sorted(label_dir.glob("*.txt")):
        classes: List[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) >= 5:
                classes.append(parts[0])
        labels[path.stem] = classes
    return labels


def build_manifest(dataset_dir: Path, output: Path, oversample_min: int = 800) -> dict:
    label_dir = dataset_dir / "labels"
    image_dir = dataset_dir / "images"
    labels = load_yolo_labels(label_dir)
    class_counts = Counter()
    for class_list in labels.values():
        class_counts.update(class_list)

    if not class_counts:
        raise ValueError("No labels found. Expected YOLO .txt files under dataset/labels.")

    manifest = []
    for stem, class_list in labels.items():
        image_path = None
        for ext in (".jpg", ".jpeg", ".png", ".bmp"):
            candidate = image_dir / f"{stem}{ext}"
            if candidate.exists():
                image_path = candidate
                break
        if image_path is None:
            continue
        weight = 1.0
        if class_list:
            rarity_scores = [1.0 / max(class_counts[c], 1) for c in class_list]
            weight = max(sum(rarity_scores), 1e-3)
        manifest.append(
            {
                "image": str(image_path.relative_to(dataset_dir)),
                "labels": class_list,
                "oversample_weight": round(weight, 6),
            }
        )

    max_count = max(class_counts.values())
    target_count = max(max_count, oversample_min)
    class_balance = {
        cls: {
            "current": count,
            "target": target_count,
            "suggested_factor": round(target_count / max(count, 1), 3),
        }
        for cls, count in class_counts.items()
    }
    payload = {
        "dataset_dir": str(dataset_dir),
        "class_counts": class_counts,
        "class_balance": class_balance,
        "samples": manifest,
        "notes": [
            "Use oversample_weight to prioritize rare classes like crack and hole.",
            "Inject hard negatives and edge cases during augmentation.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a balanced training manifest from YOLO labels.")
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--output", default=Path("outputs/balanced_manifest.json"), type=Path)
    parser.add_argument("--oversample-min", default=800, type=int)
    args = parser.parse_args()
    build_manifest(args.dataset_dir, args.output, args.oversample_min)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

