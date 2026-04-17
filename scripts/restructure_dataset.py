from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class SplitSummary:
    class_name: str
    total: int
    train: int
    val: int


def collect_images(source_root: Path) -> Dict[str, List[Path]]:
    if not source_root.exists():
        raise FileNotFoundError(f"Source root does not exist: {source_root}")
    class_map: Dict[str, List[Path]] = {}
    for class_dir in sorted(p for p in source_root.iterdir() if p.is_dir()):
        images = [
            path
            for path in sorted(class_dir.iterdir())
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if images:
            class_map[class_dir.name] = images
    if not class_map:
        raise ValueError(f"No class folders with images found under: {source_root}")
    return class_map


def ensure_clean_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_or_move(src: Path, dst: Path, move: bool) -> None:
    ensure_clean_dir(dst.parent)
    if move:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(str(src), str(dst))


def rearchitect_dataset(
    source_root: Path,
    output_root: Path,
    train_ratio: float = 0.8,
    seed: int = 42,
    move: bool = False,
) -> dict:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")
    rng = random.Random(seed)
    class_map = collect_images(source_root)
    output_images_root = output_root / "images"
    summary: List[SplitSummary] = []
    for class_name, images in class_map.items():
        shuffled = images[:]
        rng.shuffle(shuffled)
        val_count = max(1, int(round(len(shuffled) * (1.0 - train_ratio)))) if len(shuffled) > 1 else 0
        train_count = len(shuffled) - val_count
        train_images = shuffled[:train_count]
        val_images = shuffled[train_count:]

        for split_name, split_images in (("train", train_images), ("val", val_images)):
            split_dir = output_images_root / split_name / class_name
            ensure_clean_dir(split_dir)
            for idx, src in enumerate(split_images, start=1):
                dst = split_dir / f"{class_name}{idx}{src.suffix.lower()}"
                copy_or_move(src, dst, move=move)

        summary.append(
            SplitSummary(
                class_name=class_name,
                total=len(shuffled),
                train=len(train_images),
                val=len(val_images),
            )
        )

    manifest = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "train_ratio": train_ratio,
        "seed": seed,
        "mode": "move" if move else "copy",
        "splits": [summary_item.__dict__ for summary_item in summary],
    }
    ensure_clean_dir(output_root)
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Rearchitect a class-folder image dataset into train/val splits.")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("dataset/images/val/raw"),
        help="Source class-folder root, e.g. dataset/images/val/raw",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("dataset/classification"),
        help="Destination root for the restructured dataset",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Fraction of images to place in train")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed for reproducible splits")
    parser.add_argument("--move", action="store_true", help="Move files instead of copying them")
    args = parser.parse_args()

    manifest = rearchitect_dataset(
        source_root=args.source_root,
        output_root=args.output_root,
        train_ratio=args.train_ratio,
        seed=args.seed,
        move=args.move,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

